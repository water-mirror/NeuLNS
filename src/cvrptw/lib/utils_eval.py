import numpy as np
import json
import random
import math
import vrp_env
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from lib.rms import RunningMeanStd
from arguments import args
import argparse
args = args()

device = torch.device(args.device)
N_JOBS = int(args.N_JOBS)
CAP = int(args.CAP)
batch_size = int(args.BATCH)
MAX_COORD = int(args.MAX_COORD)
MAX_DIST = float(args.MAX_DIST)
LR = float(args.LR)
DEPOT_END = int(args.DEPOT_END)
SERVICE_TIME = int(args.SERVICE_TIME)
TW_WIDTH = int(args.TW_WIDTH)

N_ROLLOUT = int(args.N_ROLLOUT)
ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
N_STEPS = int(args.N_STEPS)

init_T=float(args.init_T)
final_T=float(args.final_T)

reward_norm = RunningMeanStd()

def read_input(fp):
    alpha_T = (final_T/init_T)**(1.0/N_STEPS)
    instance_data = np.load(fp)
    results = []
    for instance_index, instance in enumerate(instance_data):
        node_data = dict()
        coords = []
        for node_index, node_detail in enumerate(instance):
            x = float(node_detail[0])
            y = float(node_detail[1])
            demand = float(node_detail[2])
            ready_time = int(node_detail[3])
            due_date = int(node_detail[4])
            service_time = int(node_detail[5])
            node_data[node_index] = \
                {"x": x,
                 "y": y,
                 "demands": demand,
                 "start": ready_time,
                 "end": due_date,
                 "service_time": service_time}
            coords.append([x, y, demand])

        raw = np.array(coords)
        jobs = []
        for key, value in node_data.items():
            if key == 0:
                continue
            else:
                node_id = key-1
                node_loc = key
                node_name = str(node_id)
                x = value["x"]
                y = value["y"]
                weight = value["demands"]
                tw = {"start": value["start"], "end": value["end"]}
                service_time = value["service_time"]
                job_type = "Pickup"

                jobs.append({
                    "id": node_id,
                    "loc": node_loc,
                    "name": node_name,
                    "x": x,
                    "y": y,
                    "weight": weight,
                    "tw": tw,
                    "service_time": service_time,
                    "job_type": job_type,
                })

        dist_time = []

        def calc_dist(l,r):
            return ((l[0]-r[0])**2 + (l[1]-r[1])**2)**0.5

        for i,(x1,y1,_) in enumerate(coords):
            row = []
            for j,(x2,y2,_) in enumerate(coords):
                d = calc_dist((x1,y1),(x2,y2))
                row.append(({"dist":d,"time": d}))
            dist_time.append(row)

        adjs = []

        for i,job in enumerate(jobs):
            l = [ (j,dist_time[job['loc']][_job['loc']]['dist']) for j,_job in enumerate(jobs) ]
            l = sorted(l,key=lambda x: x[1])
            l = [x[0] for x in l]
            adjs.append(l)

        v = {
            "cap": CAP,
            "tw": {
                "start": 0,
                "end": DEPOT_END,
            },
            "start_loc": 0,
            "end_loc": 0,
            "fee_per_dist": 1.0,
            "fee_per_time": 0,
            "fixed_cost": 0,
            "handling_cost_per_weight": 0.0,
            "max_stops": 0,
            "max_dist": 0,
        }

        input_data = {
            "vehicles": [v],
            "dist_time": dist_time,
            "cost_per_absent": 1000,
            "jobs": jobs,
            "depot": coords[0][:2],
            "l_max": 10,
            "c1": 10,
            "adjs": adjs,
            "temperature": 100,
            "c2": alpha_T,
            "sa": True,
        }

        results.append([input_data, raw])
    return results

def create_env(n_jobs=99,_input=None, raw=None):

    class Env(object):
        def __init__(self,n_jobs,_input,raw):
            self.n_jobs = n_jobs
            self.input = _input
            self.raw = raw
            dist_time = _input['dist_time']
            self.dists = np.array([[ [x['dist']/MAX_DIST] for x in row ] for row in dist_time])

        def reset(self):
            self.env = vrp_env.Env(json.dumps(self.input))
            self.mapping = {}
            self.cost = 0.0
            self.best = None
            self.best_sol = None
            return self.get_states()

        def get_states(self):
# {'id': 1, 'loc': 2, 'name': '1', 'x': 92, 'y': 73, 'weight': 8, 'tw': {'start': 0, 'end': 10000}, 'service_time': 0, 'job_type': 'Pickup'}
# {'dist': 8.5440034866333, 'time': 8.5440034866333, 'stops': 1, 'time_slack': 9615.2998046875, 'wait_time': 0.0, 'service_time': 0.0, 'loc': 2, 'weight': 8.0}
            states = self.env.states()
            tours = self.env.tours()
            jobs = self.input['jobs']
            depot = self.input['depot']

            nodes = np.zeros((self.n_jobs+1,8))
            edges = np.zeros((self.n_jobs+1,self.n_jobs+1,1))
            mapping = {}

            for i,(tour,tour_state) in enumerate(zip(tours,states)):
                for j,(index,s) in enumerate(zip(tour,tour_state[1:])):
                    job = jobs[index]
                    loc = job['loc']
                    nodes[loc,:] = [job['weight']/CAP,s['weight']/CAP,s['dist']/MAX_DIST,
                                    tour_state[-1]['weight'],job['tw']['start']/MAX_DIST,
                                    job['tw']['end']/MAX_DIST,s['time']/MAX_DIST,s['time_slack']/MAX_DIST]
                    mapping[loc] = (i,j)
#                     s['time']/MAX_DIST

            for tour in tours:
                edges[0][tour[0]+1][0] = 1
                for l,r in zip(tour[0:-1],tour[1:]):
                    edges[l+1][r+1][0] = 1
                edges[tour[-1]+1][0][0] = 1

            edges = np.stack([self.dists,edges],axis=-1)
            edges = edges.reshape(-1,2)

            absents = self.env.absents()
            assert len(absents) == 0,"bad input"

            self.mapping = mapping
            self.cost = self.env.cost()
            if self.best is None or self.cost < self.best:
                self.best = self.cost
                self.best_sol = self.env.tours()

            return nodes,edges

        def sisr_step(self):
            prev_cost = self.cost
            self.env.sisr_step()
            nodes,edges = self.get_states()
            reward = prev_cost - self.cost
            return nodes,edges,reward

        def step(self,to_remove):
            prev_cost = self.cost
            self.env.step(to_remove)
            nodes,edges = self.get_states()
            reward = prev_cost - self.cost
            return nodes,edges,reward

    env = Env(n_jobs,_input, raw)
    return env

def create_batch_env(batch_size=batch_size,n_jobs=99, instance=None):

    class BatchEnv(object):
        def __init__(self,batch_size, instance):
            if instance is not None:
                one_instance = create_env(_input=instance[0], raw=instance[1])
                self.envs = [one_instance for i in range(batch_size)]
            else:
                self.envs = [ create_env(n_jobs) for i in range(batch_size) ]

        def reset(self):
            rets = [ env.reset() for env in self.envs ]
            return list(zip(*rets))

        def step(self,actions):
            actions = actions.tolist()
            assert(len(actions) == len(self.envs))
            rets = [env.step(act) for env,act in zip(self.envs,actions)]
            return list(zip(*rets))

        def sisr_step(self):
            rets = [env.sisr_step() for env in self.envs]
            return list(zip(*rets))

    return BatchEnv(batch_size, instance)

def create_replay_buffer(n_jobs=99):

    class Buffer(object):
        def __init__(self,n_jobs=n_jobs):
            super(Buffer,self).__init__()
            self.buf_nodes = []
            self.buf_edges = []
            self.buf_actions = []
            self.buf_rewards = []
            self.buf_values = []
            self.buf_log_probs = []
            self.n_jobs = n_jobs

            edges = []
            for i in range(n_jobs+1):
                for j in range(n_jobs+1):
                    edges.append([i,j])

            self.edge_index = torch.LongTensor(edges).T

        def obs(self,nodes,edges,actions,rewards,log_probs,values):
            self.buf_nodes.append(nodes)
            self.buf_edges.append(edges)
            self.buf_actions.append(actions)
            self.buf_rewards.append(rewards)
            self.buf_values.append(values)
            self.buf_log_probs.append(log_probs)

        def compute_values(self,last_v=0,_lambda = 1.0):
            rewards = np.array(self.buf_rewards)
#             rewards = (rewards - rewards.mean()) / rewards.std()
            pred_vs = np.array(self.buf_values)

            target_vs = np.zeros_like(rewards)
            advs = np.zeros_like(rewards)

#             print (rewards.shape,target_vs.shape,advs.shape,pred_vs.shape)

            v = last_v
            for i in reversed(range(rewards.shape[0])):
                v = rewards[i] + _lambda * v
                target_vs[i] = v
                adv = v - pred_vs[i]
                advs[i] = adv

            return target_vs,advs

        def gen_datas(self,last_v=0,_lambda = 1.0,batch_size=batch_size):
            target_vs,advs = self.compute_values(last_v,_lambda)
            advs = (advs - advs.mean()) / advs.std()
            l,w = target_vs.shape

            datas = []
            for i in range(l):
                for j in range(w):
                    nodes = self.buf_nodes[i][j]
                    edges = self.buf_edges[i][j]
                    action = self.buf_actions[i][j]
                    v = target_vs[i][j]
                    adv = advs[i][j]
                    log_prob = self.buf_log_probs[i][j]
#                     print (nodes.dtype,self.edge_index.dtype,edges.dtype,q,action)
                    data = Data(x=torch.from_numpy(nodes).float(),edge_index=self.edge_index,
                                edge_attr=torch.from_numpy(edges).float(),v=torch.tensor([v]).float(),
                                action=torch.tensor(action).long(),
                                log_prob=torch.tensor([log_prob]).float(),
                                adv = torch.tensor([adv]).float())
                    datas.append(data)

            return datas

        def create_data(self,_nodes,_edges):
            datas = []
            l = len(_nodes)
            for i in range(l):
                nodes = _nodes[i]
                edges = _edges[i]
                data = Data(x=torch.from_numpy(nodes).float(),edge_index=self.edge_index,edge_attr=torch.from_numpy(edges).float())
                datas.append(data)
            dl = DataLoader(datas,batch_size=l)
            return list(dl)[0]

    return Buffer()

def roll_out(model,envs,states,n_steps=10,_lambda=0.99,batch_size=batch_size,n_remove=10,is_last=False,greedy=True):
    buffer = create_replay_buffer()

    with torch.no_grad():
        model.eval()
        nodes,edges = states
        _sum = 0
        _entropy = []

        history = []
        actions_history = []
        values_history = []
        for i in range(n_steps):
            data = buffer.create_data(nodes,edges)
            data = data.to(device)
            actions,log_p,values,entropy = model(data,n_remove,greedy,1)
            actions_history.append(list(actions))
            values_history.append(envs.envs[0].env.tours())
            new_nodes,new_edges,rewards = envs.step(actions.cpu().numpy())
            rewards = np.array(rewards)
            _sum = _sum + rewards
            rewards = reward_norm(rewards)
            _entropy.append(entropy.mean().cpu().numpy())

#             buffer.obs(nodes,edges,actions.cpu().numpy(),rewards,log_p.cpu().numpy(),values.cpu().numpy())
            nodes,edges = new_nodes,new_edges
            history.append([env.cost for env in envs.envs])

        mean_value = _sum.mean()
#         print ("mean rewards:",mean_value)
#         print ("entropy:",np.mean(_entropy))
#         print ("mean cost:",np.mean([env.cost for env in envs.envs]))

        if not is_last:
#             print ("not last")
            data = buffer.create_data(nodes,edges)
            data = data.to(device)
            actions,log_p,values,entropy = model(data,n_remove,greedy)
            values = values.cpu().numpy()
        else:
            values = 0

#         dl = buffer.gen_datas(values,_lambda = _lambda,batch_size=batch_size)
        return (nodes,edges),history, np.array(actions_history), np.array(values_history)

def train_once(model,opt,dl,epoch,step,alpha=1.0):
    model.train()

    losses = []
    loss_vs = []
    loss_ps = []
    _entropy = []

    for i,batch in enumerate(dl):
        batch = batch.to(device)
        batch_size = batch.num_graphs
#         print (batch.action.shape)
        actions = batch.action.reshape((batch_size,-1))
        log_p,v,entropy = model.evaluate(batch,actions)
        _entropy.append(entropy.mean().item())

        target_vs = batch.v.squeeze(-1)
        old_log_p = batch.log_prob.squeeze(-1)
        adv =  batch.adv.squeeze(-1)

        loss_v = ((v - target_vs) ** 2).mean()

        ratio = torch.exp(log_p-old_log_p)
        obj = ratio * adv
        obj_clipped = ratio.clamp(1.0 - 0.2,
                                  1.0 + 0.2) * adv
        loss_p = -torch.min(obj, obj_clipped).mean()
        loss = loss_p + alpha * loss_v

        losses.append(loss.item())
        loss_vs.append(loss_v.item())
        loss_ps.append(loss_p.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    print ("epoch:",epoch,"step:",step,"loss_v:",np.mean(loss_vs),"loss_p:",np.mean(loss_ps),"loss:",np.mean(losses),"entropy:",np.mean(_entropy))

def eval_random(epochs,envs,n_steps=10):

    def eval_once(epoch,n_instance=128,n_steps=n_steps,batch_size=batch_size,alpha=1.0):
        nodes,edges = envs.reset()
        _sum = np.zeros(n_instance)
        for i in range(n_steps):
            actions = [ random.sample(range(0, N_JOBS),10) for i in range(n_instance) ]
            actions = np.array(actions)
            new_nodes,new_edges,rewards = envs.step(actions)
            _sum += rewards

        return np.mean([env.cost for env in envs.envs])

    print ("<<<<<<<<<<===================== random mean cost:",np.mean([eval_once(i) for i in range(epochs)]))

def random_init(envs,n_steps,n_instance=128):
    nodes,edges = envs.reset()
    for i in range(n_steps):
        actions = [ random.sample(range(0, N_JOBS),10) for i in range(n_instance) ]
        actions = np.array(actions)
        nodes,edges,rewards = envs.step(actions)

    return (nodes,edges),np.mean([env.cost for env in envs.envs])

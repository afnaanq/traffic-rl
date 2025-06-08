import cityflow
import numpy as np
import json
import os

class CityFlowEnv:
    def __init__(self, config_path, intersection_ids, reward_fn=None):
        self.eng = cityflow.Engine(config_path, thread_num=1)
        self.intersection_ids = intersection_ids
        self.action_space = [9] * len(intersection_ids)
        self.step_time = 10
        self.prev_phase = None
        self.curr_phase = None
        self.prev_total_travel_time = 0
        self.vehicle_entry_time = {}
        self.total_travel_time = 0.0
        self.prev_vehicle_ids = set()
        self.total_exited = 0

        # Reward function override
        self.reward_fn = reward_fn if reward_fn else self._compute_reward

        # Load incoming lane info
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, "..", "mydata", "incoming_lanes.json")
        with open(json_path, "r") as f:
            self.incoming_lanes = json.load(f)

        roadnet_path = os.path.join(base_dir, "..", "mydata", "roadnet.json")
        with open(roadnet_path, "r") as f:
            self.roadnet_data = json.load(f)
        self.build_lane_successor_map = self.build_lane_successor_map()

    def reset(self):
        self.eng.reset()
        return self.get_state()

    def get_state(self):
        state = []
        lane_vehicles = self.eng.get_lane_vehicle_count()  # query once per step
        for inter_id in self.intersection_ids:
            inter_state = self._extract_intersection_state(inter_id, lane_vehicles)
            state.append(inter_state)
        return np.array(state)

    def _extract_intersection_state(self, inter_id, lane_vehicles):
        # Use precomputed lane info from incoming_lanes.json
        lanes = self.incoming_lanes[inter_id]
        return [lane_vehicles.get(lane, 0) for lane in lanes]

    def set_rwd_fn(self, fn):
        self.reward_fn = fn
    
    def build_lane_successor_map(self):
        lane_successor_map = {}

        for inter in self.roadnet_data["intersections"]:
            if not inter["virtual"]:
                for road_link in inter["roadLinks"]:
                    start_road = road_link["startRoad"]
                    end_road = road_link["endRoad"]
                for lane_link in road_link["laneLinks"]:
                    in_lane = f"{start_road}_{lane_link['startLaneIndex']}"
                    out_lane = f"{start_road}_{lane_link['startLaneIndex']}"
                    lane_successor_map.setdefault(in_lane, []).append(out_lane)
        return lane_successor_map


    def step(self, actions):
        for inter_id, action in zip(self.intersection_ids, actions):
            self.eng.set_tl_phase(inter_id, action)
        
        for _ in range(self.step_time):
            self.eng.next_step()

        self.update_exited_vehicle_count()
        reward = self.reward_fn()
        state = self.get_state()
        done = self.eng.get_current_time() >= 10000  # 1 hour sim
        return state, reward, done, {}
    
    def update_exited_vehicle_count(self):
        current_vehicle_ids = set(self.eng.get_vehicles(include_waiting=True))
        exited = self.prev_vehicle_ids - current_vehicle_ids
        self.total_exited += len(exited)
        self.prev_vehicle_ids = current_vehicle_ids

        return self.total_exited

    def _compute_reward(self):
        # Get current stats
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.eng.get_lane_vehicles()
        passed_vehicles = sum(len(v) for k, v in lane_vehicles.items() if k.startswith('out'))
        total_delay = sum(lane_waiting.values())
        queue_std = np.std(list(lane_waiting.values()))

        # Emissions proxy: # of stopped vehicles (rough)
        num_stopped = sum(1 for v in lane_waiting.values() if v > 0)

        # Phase switching penalty
        switch_penalty = 1 if self.prev_phase != self.curr_phase else 0

        # Weights (tune these)
        w_passed = 0.5
        w_delay = 0.1
        w_fairness = 5.0
        w_switch = 1.0
        w_emission = 0.05

        # Final reward
        reward = (
            w_passed * passed_vehicles
            - w_delay * total_delay
            - w_fairness * queue_std
            - w_switch * switch_penalty
            - w_emission * num_stopped
        )

        # Save phase for next step
        self.prev_phase = self.curr_phase
        return reward
    
    def default_reward(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        return -sum(lane_waiting.values())
    
    def pressure_and_count_reward(self):
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()

        press_sum = 0
        for in_lane, out_lanes in self.build_lane_successor_map.items():
            q_in = lane_vehicle_count.get(in_lane, 0)
            q_out = sum(lane_vehicle_count.get(out_l, 0) for out_l in out_lanes)

            norm_factor = max(1, q_in + q_out)
            delta = max(-10, min(10, q_in - q_out))  # clip delta
            press_sum += delta / norm_factor

        waiting_sum = sum(lane_waiting.values())

        # Weighted combination
        alpha = 0.5  # pressure
        beta = 0.5   # wait count

        reward = - (alpha * abs(press_sum) * 5 + beta * waiting_sum)
        return reward


    

    def pressure_only_reward(self):

        lane_vehicle_count = self.eng.get_lane_vehicle_count()

        press_sum = 0
        for in_lane, out_lanes in self.build_lane_successor_map.items():
            q_in = lane_vehicle_count.get(in_lane, 0)
            q_out = sum(lane_vehicle_count.get(out_l, 0) for out_l in out_lanes)

            norm_factor = max(1, q_in + q_out)
            delta = max(-10, min(10, q_in - q_out))  # clip delta to [-10, 10]
            press_sum += delta / norm_factor

        reward = -abs(press_sum) * 5  # weight factor preserved from original
        return reward



    
    

    



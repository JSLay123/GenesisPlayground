#!/usr/bin/env python3
"""
Franka机械臂自动抓取固定位置cube的数据收集脚本

这个脚本实现了以下功能：
1. 获取相对位姿，并转换为绝对位姿
2. 通过逆运动学和路径规划计算插值路径，并移动到绝对位姿
3. 记录轨迹数据，并保存到文件

使用方法:
    python examples/run_collect_data.py
"""

import time
from typing import Any

import genesis as gs
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.manipulation.pick_cube_env import PickCubeEnv
from gs_env.sim.robots.config.schema import EEPoseAbsAction
from gs_env.sim.robots.manipulators import ManipulatorBase
from gs_env.sim.robots.manipulators import FrankaRobot
import json
from gs_env.common.utils.math_utils import normalize, wxyz_to_xyzw, xyzw_to_wxyz, quat_mul, quat_to_rotmat


POSE_RECORD_DIR = "pose_records"
POSE_RECORD_FILE_EXTENSION = ".json"

class AutomatedPickPlaceController:
    """自动抓取放置控制器"""
    
    def __init__(self, env: PickCubeEnv, device: torch.device = torch.device("cpu")):
        self.env = env
        self.device = device
        self.robot:FrankaRobot = env.entities["robot"]
        self.cube = env.entities["cube"]
        
        # 控制参数
        self.movement_speed = 0.01  # 移动速度
        self.gripper_open_width = 0.04  # 张开夹爪宽度
        self.gripper_close_width = 0.0  # 闭合夹爪宽度
        self.approach_height = 0.15  # 接近高度
        self.grasp_height = 0.05  # 抓取高度
        self.lift_height = 0.2  # 提升高度
        
        # 目标位置（固定cube位置）
        self.cube_target_pos = torch.tensor([0.3, 0.0, 0.05], device=device)
        self.place_target_pos = torch.tensor([0.5, 0.0, 0.05], device=device)

        # 目标四元数
        self.cube_target_quat = torch.tensor([1.0, 0.3, 0.0, 0.0], device=device)
        self.place_target_quat = torch.tensor([1.0, 0.6, 0.0, 0.0], device=device)
        
        # 默认末端执行器方向（垂直向下）
        self.default_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # w, x, y, z

        # 读取pose_records
        self.pose_records = json.load(open("pose_records/pose_record_1758453948.json"))

    def get_current_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """获取当前末端执行器位置和方向"""
        ee_pose = self.robot.ee_pose
        position = ee_pose[..., :3]
        orientation = ee_pose[..., 3:]
        return position, orientation

    
    def move_to_position(self, target_pos: torch.Tensor, target_quat: torch.Tensor, 
                        gripper_width: float, steps: int = 50) -> None:
        """平滑移动到目标位置"""
        self.robot._apply_ee_pose_abs_path_planner(EEPoseAbsAction(
            ee_link_pos=target_pos,
            ee_link_quat=target_quat,
            gripper_width=gripper_width
        ))

    
    def _quaternion_slerp(self, start_quat: torch.Tensor, end_quat: torch.Tensor, 
                         steps: int) -> torch.Tensor:
        """四元数球面线性插值"""
        # 简化的线性插值（对于小角度变化足够）
        result = torch.zeros((steps, 4), device=self.device)
        for i in range(steps):
            t = (i + 1) / steps
            result[i] = (1 - t) * start_quat + t * end_quat
            result[i] = result[i] / torch.norm(result[i])  # 归一化
        return result
    
    def wait_for_stabilization(self, steps: int = 10) -> None:
        """等待系统稳定"""
        for _ in range(steps):
            self.env.scene.step()
            time.sleep(0.01)

    
    def relative_to_absolute_pose(self, pos_ref:torch.Tensor, quat_ref:torch.Tensor, 
                                            pos_rel:torch.Tensor, quat_rel:torch.Tensor):
        """
        将相对位姿转换为绝对位姿
        输入: pos_ref (N,3), quat_ref (N,4), pos_rel (N,3), quat_rel (N,4)
        输出: pos_abs (N,3), quat_abs (N,4)
        """
        # normalize quat_ref
        quat_ref = normalize(quat_ref)
        # convert to rotation matrix
        rot_ref = quat_to_rotmat(quat_ref)
        pos_abs = torch.bmm(rot_ref, pos_rel.unsqueeze(-1)).squeeze(-1) + pos_ref
        quat_abs = quat_mul(quat_ref, quat_rel)
        return pos_abs, normalize(quat_abs)
    
    def execute_pick_place_sequence(self) -> dict[str, Any]:
        """执行完整的抓取放置序列"""
        print("🤖 开始执行自动抓取放置序列...")
        
        # 记录轨迹数据
        trajectory_data = {
            "actions": [],
            "observations": [],
            "timestamps": [],
            "metadata": {
                "task": "automated_pick_place",
                "cube_position": self.cube_target_pos.cpu().numpy().tolist(),
                "place_position": self.place_target_pos.cpu().numpy().tolist(),
                "timestamp": time.time()
            }
        }
        
        try:
            # 1. 移动到cube上方
            print("📍 步骤1: 移动到cube上方...")
            # 从pose_records中获取相对位姿，并转换为绝对位姿
            n = self.env.num_envs  # 并行环境数量
            relative_pos = torch.tensor(self.pose_records["relative_position"], device=self.device).unsqueeze(0).repeat(n, 1)  # (n, 3)
            relative_quat = torch.tensor(self.pose_records["relative_orientation"], device=self.device).unsqueeze(0).repeat(n, 1)  # (n, 4)

            cube_pos, cube_quat = self.cube.get_pos(), self.cube.get_quat()

            approach_pos, approach_quat = self.relative_to_absolute_pose(cube_pos, cube_quat, 
                                                                        relative_pos, relative_quat)
            print(f"🤖 移动到cube上方位姿: {approach_pos}, {approach_quat}")                                                                         
            self.move_to_position(approach_pos, approach_quat, self.gripper_open_width)
            self._record_step(trajectory_data, "approach_cube")
            
            # # 2. 下降到抓取位置
            # print("📉 步骤2: 下降到抓取位置...")
            # grasp_pos = self.cube_target_pos.clone()
            # grasp_pos[2] = self.grasp_height
            # self.move_to_position(grasp_pos, self.default_orientation, self.gripper_open_width)
            # self._record_step(trajectory_data, "descend_to_grasp")
            
            # # 3. 闭合夹爪抓取cube
            # print("🤏 步骤3: 抓取cube...")
            # self.move_to_position(grasp_pos, self.default_orientation, self.gripper_close_width)
            # self.wait_for_stabilization()  # 等待抓取稳定
            # self._record_step(trajectory_data, "grasp_cube")
            
            # # 4. 提升cube
            # print("📈 步骤4: 提升cube...")
            # lift_pos = self.cube_target_pos.clone()
            # lift_pos[2] = self.lift_height
            # self.move_to_position(lift_pos, self.default_orientation, self.gripper_close_width)
            # self._record_step(trajectory_data, "lift_cube")
            
            # # 5. 移动到放置位置上方
            # print("🚚 步骤5: 移动到放置位置上方...")
            # place_approach_pos = self.place_target_pos.clone()
            # place_approach_pos[2] = self.lift_height
            # self.move_to_position(place_approach_pos, self.default_orientation, self.gripper_close_width)
            # self._record_step(trajectory_data, "move_to_place")
            
            # # 6. 下降到放置位置
            # print("📉 步骤6: 下降到放置位置...")
            # place_pos = self.place_target_pos.clone()
            # place_pos[2] = self.grasp_height
            # self.move_to_position(place_pos, self.default_orientation, self.gripper_close_width)
            # self._record_step(trajectory_data, "descend_to_place")
            
            # # 7. 张开夹爪放置cube
            # print("🤲 步骤7: 放置cube...")
            # self.move_to_position(place_pos, self.default_orientation, self.gripper_open_width)
            # self._record_step(trajectory_data, "place_cube")
            
            # # 8. 提升到安全高度
            # print("📈 步骤8: 提升到安全高度...")
            # safe_pos = self.place_target_pos.clone()
            # safe_pos[2] = self.lift_height
            # self.move_to_position(safe_pos, self.default_orientation, self.gripper_open_width)
            # self._record_step(trajectory_data, "lift_to_safe")
            
            # print("✅ 抓取放置序列执行完成!")
            # return trajectory_data
            
        except Exception as e:
            print(f"❌ 执行过程中出现错误: {e}")
            return trajectory_data
    
    def _record_step(self, trajectory_data: dict[str, Any], step_name: str) -> None:
        """记录当前步骤的数据"""
        # 获取当前观察
        obs = self.env.get_observations()
        
        # 获取当前末端执行器位置
        ee_pos, ee_quat = self.get_current_ee_pose()
        
        # 记录数据
        trajectory_data["actions"].append({
            "step_name": step_name,
            "ee_position": ee_pos.cpu().numpy().tolist(),
            "ee_orientation": ee_quat.cpu().numpy().tolist(),
        })
        
        trajectory_data["observations"].append(obs.cpu().numpy().tolist())
        trajectory_data["timestamps"].append(time.time())
        
        print(f"  📝 记录步骤: {step_name}")


def create_gs_env(env_name: str = "pick_cube_default") -> PickCubeEnv:
    """创建Genesis环境"""
    device = torch.device("cuda")
    return PickCubeEnv(args=EnvArgsRegistry[env_name], device=device)


def set_fixed_cube_position(env: PickCubeEnv, position: tuple[float, float, float], quat: tuple[float, float, float, float]) -> None:
    """设置cube到固定位置"""
    cube_pos = torch.tensor(position, device=env.device)
    cube_quat = torch.tensor(quat, device=env.device)  # 无旋转
    
    env.entities["cube"].set_pos(cube_pos)
    env.entities["cube"].set_quat(cube_quat)
    
    print(f"🎯 Cube位置设置为: {position}")
    print(f"🎯 Cube四元数设置为: {quat}")


def save_trajectory_data(trajectory_data: dict[str, Any], filename: str | None = None) -> str:
    """保存轨迹数据到文件"""
    import pickle
    import os
    
    if filename is None:
        timestamp = int(time.time())
        filename = f"automated_pick_place_{timestamp}.pkl"
    
    # 确保目录存在
    os.makedirs("trajectories", exist_ok=True)
    filepath = os.path.join("trajectories", filename)
    
    with open(filepath, "wb") as f:
        pickle.dump(trajectory_data, f)
    
    print(f"💾 轨迹数据已保存到: {filepath}")
    return filepath


def main() -> None:
    """主函数"""
    print("🚀 初始化Franka自动抓取数据收集系统...")
    
    # 初始化Genesis
    gs.init(
        seed=0,
        precision="32",
        logging_level="info",
        backend=gs.gpu,  # type: ignore
    )
    
    try:
        # 创建环境
        print("🏗️ 创建环境...")
        env = create_gs_env()   
        
        # 创建自动控制器
        print("🤖 创建自动控制器...")
        controller = AutomatedPickPlaceController(env, device=torch.device("cuda"))
        
        # 执行抓取放置序列
        print("\n" + "=" * 60)
        print("🎬 开始执行自动抓取放置任务")
        print("=" * 60)
        
        trajectory_data = controller.execute_pick_place_sequence()
        
        # 保存轨迹数据
        print("\n💾 保存轨迹数据...")
        save_trajectory_data(trajectory_data)
        
        print("\n✅ 数据收集完成!")
        print("📊 收集的数据包括:")
        print(f"   - 动作数量: {len(trajectory_data['actions'])}")
        print(f"   - 观察数量: {len(trajectory_data['observations'])}")
        print(f"   - 时间戳数量: {len(trajectory_data['timestamps'])}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 清理资源...")


if __name__ == "__main__":
    main()

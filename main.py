import sys
from typing import List, Tuple, Dict, Optional

# ===================== 数据结构定义 完全匹配文档 =====================
class Server:
    def __init__(self, sid: int, g_total: int, vg_single: int, c_total: int, r_total: int):
        self.sid = sid          # 服务器编号 1~M
        self.g_total = g_total  # GPU总数量 G_s
        self.vg_single = vg_single  # 单卡显存 VG_s
        self.c_total = c_total  # CPU总核数 C_s
        self.r_total = r_total  # 总内存 R_s
        # key:离散时间t, value:(已占用GPU,已占用CPU,已占用内存)
        self.used: Dict[int, Tuple[int, int, int]] = {}

    # 获取指定时间t空闲资源
    def get_free(self, t: int) -> Tuple[int, int, int]:
        g_used, c_used, m_used = self.used.get(t, (0, 0, 0))
        free_g = self.g_total - g_used
        free_c = self.c_total - c_used
        free_m = self.r_total - m_used
        return free_g, free_c, free_m

    # 占用连续时间片 [t_start, t_end] 资源
    def occupy_resource(self, t_start: int, t_end: int, u_gpu: int, req_c: int, req_m: int):
        for t in range(t_start, t_end + 1):
            g_old, c_old, m_old = self.used.get(t, (0, 0, 0))
            self.used[t] = (g_old + u_gpu, c_old + req_c, m_old + req_m)


class Job:
    def __init__(self, jid: int, release_t: int, proc_len: int, g_req: int, v_req: int, c_req: int, m_req: int, weight: int):
        # 任务原始需求 Job_i = (r_i,p_i,g_i,v_i,c_i,m_i,w_i)
        self.jid = jid
        self.release_t = release_t
        self.proc_len = proc_len
        self.g_req = g_req
        self.v_req = v_req
        self.c_req = c_req
        self.m_req = m_req
        self.weight = weight
        # 调度输出 Schedule_i = (i,s_i,t_i,u_i,F_i)
        self.sid: int = -1
        self.t_start: int = -1
        self.u: int = -1
        self.f_finish: int = -1

# ===================== 输入解析模块 严格匹配文档输入格式 =====================
def read_standard_input() -> Tuple[List[Server], List[Job]]:
    all_nums = list(map(int, sys.stdin.read().split()))
    ptr = 0
    server_cnt = all_nums[ptr]
    job_cnt = all_nums[ptr + 1]
    ptr += 2

    server_list: List[Server] = []
    for sid in range(1, server_cnt + 1):
        g_total = all_nums[ptr]
        vg_single = all_nums[ptr + 1]
        c_total = all_nums[ptr + 2]
        r_total = all_nums[ptr + 3]
        ptr += 4
        server_list.append(Server(sid, g_total, vg_single, c_total, r_total))

    job_list: List[Job] = []
    for jid in range(1, job_cnt + 1):
        release_t = all_nums[ptr]
        proc_len = all_nums[ptr + 1]
        g_req = all_nums[ptr + 2]
        v_req = all_nums[ptr + 3]
        c_req = all_nums[ptr + 4]
        m_req = all_nums[ptr + 5]
        weight = all_nums[ptr + 6]
        ptr += 7
        job_list.append(Job(jid, release_t, proc_len, g_req, v_req, c_req, m_req, weight))
    return server_list, job_list

# ===================== 调度辅助工具函数 =====================
# 判断服务器能否承载任务，返回满足条件的最小u；不兼容返回None
def get_min_valid_u(server: Server, job: Job) -> Optional[int]:
    # CPU、内存单台上限校验
    if job.c_req > server.c_total or job.m_req > server.r_total:
        return None
    # 遍历合法u区间 [g, G]，取最小满足显存需求
    for u_candidate in range(job.g_req, server.g_total + 1):
        if job.v_req <= u_candidate * server.vg_single:
            return u_candidate
    return None

# 查找服务器上从earliest_t开始，连续duration时长全部空闲的最早起始时间
def find_continuous_time_window(
    s: Server,
    req_u: int,
    req_c: int,
    req_m: int,
    earliest_t: int,
    duration: int
) -> int:
    current_t = earliest_t
    while True:
        window_ok = True
        for delta in range(duration):
            t = current_t + delta
            fg, fc, fm = s.get_free(t)
            if fg < req_u or fc < req_c or fm < req_m:
                window_ok = False
                break
        if window_ok:
            return current_t
        current_t += 1

# ===================== 核心调度算法（贪心：高优先级任务优先） =====================
def run_scheduler(servers: List[Server], jobs: List[Job]):
    # 按优先级权重降序，高weight任务先分配
    sorted_jobs = sorted(jobs, key=lambda x: -x.weight)
    for job in sorted_jobs:
        best_start_t = float('inf')
        best_server: Optional[Server] = None
        best_u = -1

        # 遍历所有服务器，寻找最优部署机器
        for s in servers:
            valid_u = get_min_valid_u(s, job)
            if valid_u is None:
                continue
            # 寻找最早可用启动时间
            start_t = find_continuous_time_window(s, valid_u, job.c_req, job.m_req, job.release_t, job.proc_len)
            if start_t < best_start_t:
                best_start_t = start_t
                best_server = s
                best_u = valid_u

        # 分配调度结果
        job.sid = best_server.sid
        job.t_start = best_start_t
        job.u = best_u
        job.f_finish = best_start_t + job.proc_len
        # 修复报错：原代码t_start未定义，替换为best_start_t
        t_end = best_start_t + job.proc_len - 1
        best_server.occupy_resource(best_start_t, t_end, best_u, job.c_req, job.m_req)

# ===================== 全局约束校验函数（对标评测合法性检测） =====================
def check_all_constraints(servers: List[Server], jobs: List[Job]) -> bool:
    # 约束1：任务无重复、无遗漏
    jid_set = set()
    for j in jobs:
        if j.jid in jid_set:
            return False
        jid_set.add(j.jid)
    if len(jid_set) != len(jobs):
        return False

    for job in jobs:
        # 找到对应服务器
        target_s = None
        for s in servers:
            if s.sid == job.sid:
                target_s = s
                break
        if target_s is None:
            return False

        # 约束2：t_i >= r_i
        if job.t_start < job.release_t:
            return False

        # 约束4：F_i = t_i + p_i
        if job.f_finish != job.t_start + job.proc_len:
            return False

        # 约束5：u范围、显存、单机CPU/内存上限
        if not (job.g_req <= job.u <= target_s.g_total):
            return False
        if job.v_req > job.u * target_s.vg_single:
            return False
        if job.c_req > target_s.c_total or job.m_req > target_s.r_total:
            return False

        # 约束3、6：连续运行、服务器并发资源不超限
        t_end = job.t_start + job.proc_len - 1
        for t in range(job.t_start, t_end + 1):
            g_use, c_use, m_use = target_s.used[t]
            if g_use > target_s.g_total or c_use > target_s.c_total or m_use > target_s.r_total:
                return False
    return True

# ===================== 标准输出模块（严格匹配输出格式） =====================
def output_result(jobs: List[Job]):
    # 按任务编号升序输出，无多余文字、空行
    jobs_sorted = sorted(jobs, key=lambda x: x.jid)
    for j in jobs_sorted:
        print(f"{j.jid} {j.sid} {j.t_start} {j.u} {j.f_finish}")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    server_arr, job_arr = read_standard_input()
    run_scheduler(server_arr, job_arr)
    # 合法性校验，非法直接抛出异常
    if not check_all_constraints(server_arr, job_arr):
        raise RuntimeError("Generated schedule violates constraints!")
    output_result(job_arr)
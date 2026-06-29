import sys
from bisect import bisect_right
from typing import List, Tuple, Optional, Dict


# 正式提交时保持 False，避免产生额外耗时或输出
DEBUG = False


class Server:
    """
    使用区间资源表维护服务器占用情况。

    blocks 中的每个元素为：
        [start, end, used_gpu, used_cpu, used_mem]

    表示半开区间 [start, end) 内服务器资源占用恒定。
    任务运行区间 [t_i, t_i + p_i - 1] 等价于半开区间 [t_i, t_i + p_i)。
    """
    #限制服务器只能有这些属性
    __slots__ = (
        "sid",
        "g_total",
        "vg_single",
        "c_total",
        "r_total",
        "blocks",
        "ends",
    )

    def __init__(self, sid: int, g_total: int, vg_single: int, c_total: int, r_total: int):
        self.sid = sid   #服务器编号
        self.g_total = g_total  #服务器有多少gpu
        self.vg_single = vg_single #gpu显存大小
        self.c_total = c_total  #cpu核数
        self.r_total = r_total  #内存总量

        # 有资源占用的区间。没有出现在 blocks 中的时间段，表示资源占用为 0。
        self.blocks: List[List[int]] = []

        # blocks 的 end 列，用于二分快速定位第一个 end > t 的区间。
        self.ends: List[int] = []

    def _rebuild_ends(self) -> None:
        self.ends = [b[1] for b in self.blocks]

    def find_window(
        self,
        req_gpu: int,
        req_cpu: int,
        req_mem: int,
        earliest_t: int,
        duration: int,
    ) -> int:
        """
        找到从 earliest_t 开始，能够连续运行 duration 个时间单位的最早开始时间。

        关键优化：
        若某个已有占用区间资源不足，则当前窗口以及所有与该不足区间相交的起点都不可能合法，
        因此直接跳到该不足区间的 end，而不是 current_t += 1。
        """

        t = earliest_t

        blocks = self.blocks
        ends = self.ends
        g_total = self.g_total
        c_total = self.c_total
        r_total = self.r_total

        while True:
            need_end = t + duration

            # 找到第一个 end > t 的区间
            idx = bisect_right(ends, t)

            ok = True

            while idx < len(blocks):
                b_start, b_end, used_g, used_c, used_m = blocks[idx]

                # 后续区间已经在目标窗口之后，不影响当前窗口
                if b_start >= need_end:
                    break

                # 理论上 bisect 后不会出现，但保留用于稳妥
                if b_end <= t:
                    idx += 1
                    continue

                # 当前已有占用区间与 [t, need_end) 相交，检查资源是否够
                if (
                    used_g + req_gpu > g_total
                    or used_c + req_cpu > c_total
                    or used_m + req_mem > r_total
                ):
                    # 跳过整个资源不足区间
                    t = b_end
                    ok = False
                    break

                idx += 1

            if ok:
                return t

    def occupy_resource(
        self,
        start: int,
        end: int,
        add_gpu: int,
        add_cpu: int,
        add_mem: int,
    ) -> None:
        """
        在半开区间 [start, end) 上增加资源占用。
        """

        old_blocks = self.blocks
        new_blocks: List[List[int]] = []

        cur = start

        for block in old_blocks:
            b_start, b_end, used_g, used_c, used_m = block

            # 当前旧区间完全在新增区间左侧
            if b_end <= start:
                new_blocks.append(block)
                continue

            # 当前旧区间完全在新增区间右侧
            if b_start >= end:
                if cur < end:
                    new_blocks.append([cur, end, add_gpu, add_cpu, add_mem])
                    cur = end

                new_blocks.append(block)
                continue

            # 当前旧区间与新增区间有交集

            # 旧区间左侧未被新增区间覆盖的部分，保持原占用
            if b_start < start:
                new_blocks.append([b_start, start, used_g, used_c, used_m])

            overlap_left = max(b_start, start)

            # 如果新增区间中有一段落在空白区间，补上新增占用
            if cur < overlap_left:
                new_blocks.append([cur, overlap_left, add_gpu, add_cpu, add_mem])
                cur = overlap_left

            overlap_right = min(b_end, end)

            # 交集部分叠加占用
            if overlap_left < overlap_right:
                new_blocks.append(
                    [
                        overlap_left,
                        overlap_right,
                        used_g + add_gpu,
                        used_c + add_cpu,
                        used_m + add_mem,
                    ]
                )
                cur = overlap_right

            # 旧区间右侧未被新增区间覆盖的部分，保持原占用
            if b_end > end:
                if cur < end:
                    new_blocks.append([cur, end, add_gpu, add_cpu, add_mem])
                    cur = end

                new_blocks.append([end, b_end, used_g, used_c, used_m])
                cur = end

        # 新增区间右侧剩余部分
        if cur < end:
            new_blocks.append([cur, end, add_gpu, add_cpu, add_mem])

        # 合并相邻且资源占用完全相同的区间，减少 blocks 数量
        merged: List[List[int]] = []

        for b in new_blocks:
            if b[0] >= b[1]:
                continue

            if (
                merged
                and merged[-1][1] == b[0]
                and merged[-1][2] == b[2]
                and merged[-1][3] == b[3]
                and merged[-1][4] == b[4]
            ):
                merged[-1][1] = b[1]
            else:
                merged.append(b)

        self.blocks = merged
        self._rebuild_ends()


class Job:
    __slots__ = (
        "jid",
        "release_t",
        "proc_len",
        "g_req",
        "v_req",
        "c_req",
        "m_req",
        "weight",
        "sid",
        "t_start",
        "u",
        "f_finish",
        "candidates",
        "compat_cnt",
    )

    def __init__(
        self,
        jid: int,
        release_t: int,
        proc_len: int,
        g_req: int,
        v_req: int,
        c_req: int,
        m_req: int,
        weight: int,
    ):
        self.jid = jid
        self.release_t = release_t
        self.proc_len = proc_len
        self.g_req = g_req
        self.v_req = v_req
        self.c_req = c_req
        self.m_req = m_req
        self.weight = weight

        # 调度结果
        self.sid = -1
        self.t_start = -1
        self.u = -1
        self.f_finish = -1

        # 候选服务器：
        # (server, valid_u, memory_waste, server_gpu_memory, cpu_waste, mem_waste)
        self.candidates: List[Tuple[Server, int, int, int, int, int]] = []
        self.compat_cnt = 0


def read_standard_input() -> Tuple[List[Server], List[Job]]:
    data = list(map(int, sys.stdin.buffer.read().split()))

    if not data:
        return [], []

    ptr = 0

    server_cnt = data[ptr]
    job_cnt = data[ptr + 1]
    ptr += 2

    servers: List[Server] = []

    for sid in range(1, server_cnt + 1):
        g_total = data[ptr]
        vg_single = data[ptr + 1]
        c_total = data[ptr + 2]
        r_total = data[ptr + 3]
        ptr += 4

        servers.append(Server(sid, g_total, vg_single, c_total, r_total))

    jobs: List[Job] = []

    for jid in range(1, job_cnt + 1):
        release_t = data[ptr]
        proc_len = data[ptr + 1]
        g_req = data[ptr + 2]
        v_req = data[ptr + 3]
        c_req = data[ptr + 4]
        m_req = data[ptr + 5]
        weight = data[ptr + 6]
        ptr += 7

        jobs.append(
            Job(
                jid,
                release_t,
                proc_len,
                g_req,
                v_req,
                c_req,
                m_req,
                weight,
            )
        )

    return servers, jobs


def get_min_valid_u(server: Server, job: Job) -> Optional[int]:
    """
    计算任务在某服务器上的最小合法 GPU 分配数量 u。

    约束：
        u >= g_i
        u <= G_s
        v_i <= u * VG_s
        c_i <= C_s
        m_i <= R_s
    """

    if job.c_req > server.c_total or job.m_req > server.r_total:
        return None

    # 满足显存需求需要的最少 GPU 数
    u_by_memory = (job.v_req + server.vg_single - 1) // server.vg_single

    valid_u = max(job.g_req, u_by_memory)

    if valid_u <= server.g_total:
        return valid_u

    return None


def prepare_candidates(servers: List[Server], jobs: List[Job]) -> None:
    """
    预处理每个任务能够使用的服务器。
    任务与服务器是否兼容只与资源上限有关，与时间无关，因此提前计算。
    """

    for job in jobs:
        candidates: List[Tuple[Server, int, int, int, int, int]] = []

        for server in servers:
            valid_u = get_min_valid_u(server, job)

            if valid_u is None:
                continue

            memory_waste = valid_u * server.vg_single - job.v_req
            server_gpu_memory = server.g_total * server.vg_single
            cpu_waste = server.c_total - job.c_req
            mem_waste = server.r_total - job.m_req

            candidates.append(
                (
                    server,
                    valid_u,
                    memory_waste,
                    server_gpu_memory,
                    cpu_waste,
                    mem_waste,
                )
            )

        # 候选服务器排序：
        # 1. 显存浪费少
        # 2. 使用 GPU 数少
        # 3. CPU 浪费少
        # 4. 内存浪费少
        # 5. 服务器 GPU 总显存小，尽量把小任务放小机器，保留大机器
        # 6. 服务器编号小，保证结果稳定
        candidates.sort(
            key=lambda x: (
                x[2],
                x[1],
                x[4],
                x[5],
                x[3],
                x[0].sid,
            )
        )

        job.candidates = candidates
        job.compat_cnt = len(candidates)


def job_sort_key(job: Job) -> Tuple[int, int, int, int, int, int]:
    """
    任务排序策略。

    主要思想：
    1. 先安排兼容服务器少的瓶颈任务，避免后续无处可放；
    2. 单位时间权重高的任务优先，降低加权等待；
    3. 提交时间早的任务优先；
    4. 权重大、运行时间长的任务作为后续 tie-break。
    """

    # 避免浮点数，扩大 1000000 倍后做整数比较
    weight_density = job.weight * 1000000 // job.proc_len

    return (
        job.compat_cnt,
        -weight_density,
        job.release_t,
        -job.weight,
        -job.proc_len,
        job.jid,
    )


def run_scheduler(servers: List[Server], jobs: List[Job]) -> None:
    if not jobs:
        return

    prepare_candidates(servers, jobs)

    sorted_jobs = sorted(jobs, key=job_sort_key)

    for job in sorted_jobs:
        best_key = None
        best_server: Optional[Server] = None
        best_u = -1
        best_start_t = -1

        for (
            server,
            valid_u,
            memory_waste,
            server_gpu_memory,
            cpu_waste,
            mem_waste,
        ) in job.candidates:

            start_t = server.find_window(
                valid_u,
                job.c_req,
                job.m_req,
                job.release_t,
                job.proc_len,
            )

            candidate_key = (
                start_t,            # 第一目标：开始越早越好
                memory_waste,       # 第二目标：显存碎片越少越好
                valid_u,            # 第三目标：占用 GPU 数越少越好
                cpu_waste,          # 第四目标：CPU 资源越贴合越好
                mem_waste,          # 第五目标：内存资源越贴合越好
                server_gpu_memory,  # 第六目标：尽量保留大显存机器
                server.sid,         # 稳定输出
            )

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_server = server
                best_u = valid_u
                best_start_t = start_t

            # 如果已经能在 release_t 启动，由于候选服务器已按碎片度排序，
            # 后面的服务器不可能获得更早开始时间，因此可以提前结束搜索。
            if start_t == job.release_t:
                break

        # 输入保证每个任务至少可以在某台服务器上单独运行，因此 best_server 正常不会为 None。
        if best_server is None:
            raise RuntimeError("No feasible server found for job {}".format(job.jid))

        job.sid = best_server.sid
        job.t_start = best_start_t
        job.u = best_u
        job.f_finish = best_start_t + job.proc_len

        best_server.occupy_resource(
            best_start_t,
            best_start_t + job.proc_len,
            best_u,
            job.c_req,
            job.m_req,
        )


def check_all_constraints(servers: List[Server], jobs: List[Job]) -> bool:
    """
    本地调试用合法性检查。
    正式提交时 DEBUG=False，不会执行此函数。
    """

    if not jobs:
        return True

    server_map: Dict[int, Server] = {s.sid: s for s in servers}

    seen = set()

    # 用事件扫描检查每台服务器的并发资源是否超限
    events: Dict[int, List[Tuple[int, int, int, int]]] = {
        s.sid: [] for s in servers
    }

    for job in jobs:
        if job.jid in seen:
            return False
        seen.add(job.jid)

        server = server_map.get(job.sid)

        if server is None:
            return False

        if job.t_start < job.release_t:
            return False

        if job.f_finish != job.t_start + job.proc_len:
            return False

        if not (job.g_req <= job.u <= server.g_total):
            return False

        if job.v_req > job.u * server.vg_single:
            return False

        if job.c_req > server.c_total or job.m_req > server.r_total:
            return False

        events[job.sid].append((job.t_start, job.u, job.c_req, job.m_req))
        events[job.sid].append((job.f_finish, -job.u, -job.c_req, -job.m_req))

    if len(seen) != len(jobs):
        return False

    for server in servers:
        server_events = events[server.sid]
        server_events.sort()

        used_g = 0
        used_c = 0
        used_m = 0

        idx = 0

        while idx < len(server_events):
            current_t = server_events[idx][0]

            # 同一时刻的开始和结束事件一起处理。
            # 因为区间是 [start, finish)，finish 时刻资源已经释放。
            while idx < len(server_events) and server_events[idx][0] == current_t:
                _, delta_g, delta_c, delta_m = server_events[idx]
                used_g += delta_g
                used_c += delta_c
                used_m += delta_m
                idx += 1

            if used_g < 0 or used_c < 0 or used_m < 0:
                return False

            if (
                used_g > server.g_total
                or used_c > server.c_total
                or used_m > server.r_total
            ):
                return False

    return True


def output_result(jobs: List[Job]) -> None:
    jobs_sorted = sorted(jobs, key=lambda x: x.jid)

    out_lines = []

    for job in jobs_sorted:
        out_lines.append(
            "{} {} {} {} {}".format(
                job.jid,
                job.sid,
                job.t_start,
                job.u,
                job.f_finish,
            )
        )

    sys.stdout.write("\n".join(out_lines))


def main() -> None:
    servers, jobs = read_standard_input()

    if not servers and not jobs:
        return

    run_scheduler(servers, jobs)

    if DEBUG:
        if not check_all_constraints(servers, jobs):
            raise RuntimeError("Generated schedule violates constraints!")

    output_result(jobs)


if __name__ == "__main__":
    main()
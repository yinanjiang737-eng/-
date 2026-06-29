#include <bits/stdc++.h>
using namespace std;

static const bool DEBUG_MODE = false;

struct Block {
    int start;
    int end;
    int used_g;
    int used_c;
    int used_m;
};

struct Server {
    int sid;
    int g_total;
    int vg_single;
    int c_total;
    int r_total;

    vector<Block> blocks;
    vector<int> ends;

    Server() = default;

    Server(int sid_, int g_total_, int vg_single_, int c_total_, int r_total_)
        : sid(sid_),
          g_total(g_total_),
          vg_single(vg_single_),
          c_total(c_total_),
          r_total(r_total_) {}

    void rebuild_ends() {
        ends.clear();
        ends.reserve(blocks.size());
        for (const auto &b : blocks) {
            ends.push_back(b.end);
        }
    }

    int find_window(int req_gpu, int req_cpu, int req_mem, int earliest_t, int duration) const {
        int t = earliest_t;

        while (true) {
            int need_end = t + duration;

            int idx = upper_bound(ends.begin(), ends.end(), t) - ends.begin();
            bool ok = true;

            while (idx < (int)blocks.size()) {
                const Block &b = blocks[idx];

                if (b.start >= need_end) {
                    break;
                }

                if (b.end <= t) {
                    ++idx;
                    continue;
                }

                if (b.used_g + req_gpu > g_total ||
                    b.used_c + req_cpu > c_total ||
                    b.used_m + req_mem > r_total) {
                    t = b.end;
                    ok = false;
                    break;
                }

                ++idx;
            }

            if (ok) {
                return t;
            }
        }
    }

    void occupy_resource(int start, int end, int add_gpu, int add_cpu, int add_mem) {
        vector<Block> new_blocks;
        new_blocks.reserve(blocks.size() + 3);

        int cur = start;

        for (const auto &b : blocks) {
            int b_start = b.start;
            int b_end = b.end;
            int used_g = b.used_g;
            int used_c = b.used_c;
            int used_m = b.used_m;

            if (b_end <= start) {
                new_blocks.push_back(b);
                continue;
            }

            if (b_start >= end) {
                if (cur < end) {
                    new_blocks.push_back({cur, end, add_gpu, add_cpu, add_mem});
                    cur = end;
                }
                new_blocks.push_back(b);
                continue;
            }

            if (b_start < start) {
                new_blocks.push_back({b_start, start, used_g, used_c, used_m});
            }

            int overlap_left = max(b_start, start);

            if (cur < overlap_left) {
                new_blocks.push_back({cur, overlap_left, add_gpu, add_cpu, add_mem});
                cur = overlap_left;
            }

            int overlap_right = min(b_end, end);

            if (overlap_left < overlap_right) {
                new_blocks.push_back({
                    overlap_left,
                    overlap_right,
                    used_g + add_gpu,
                    used_c + add_cpu,
                    used_m + add_mem
                });
                cur = overlap_right;
            }

            if (b_end > end) {
                if (cur < end) {
                    new_blocks.push_back({cur, end, add_gpu, add_cpu, add_mem});
                    cur = end;
                }
                new_blocks.push_back({end, b_end, used_g, used_c, used_m});
                cur = end;
            }
        }

        if (cur < end) {
            new_blocks.push_back({cur, end, add_gpu, add_cpu, add_mem});
        }

        vector<Block> merged;
        merged.reserve(new_blocks.size());

        for (const auto &b : new_blocks) {
            if (b.start >= b.end) {
                continue;
            }

            if (!merged.empty() &&
                merged.back().end == b.start &&
                merged.back().used_g == b.used_g &&
                merged.back().used_c == b.used_c &&
                merged.back().used_m == b.used_m) {
                merged.back().end = b.end;
            } else {
                merged.push_back(b);
            }
        }

        blocks.swap(merged);
        rebuild_ends();
    }
};

struct Candidate {
    int server_idx;
    int valid_u;
    int memory_waste;
    int server_gpu_memory;
    int cpu_waste;
    int mem_waste;
};

struct Job {
    int jid;
    int release_t;
    int proc_len;
    int g_req;
    int v_req;
    int c_req;
    int m_req;
    int weight;

    int sid = -1;
    int t_start = -1;
    int u = -1;
    int f_finish = -1;

    vector<Candidate> candidates;
    int compat_cnt = 0;
};

int get_min_valid_u(const Server &server, const Job &job) {
    if (job.c_req > server.c_total || job.m_req > server.r_total) {
        return -1;
    }

    int u_by_memory = (job.v_req + server.vg_single - 1) / server.vg_single;
    int valid_u = max(job.g_req, u_by_memory);

    if (valid_u <= server.g_total) {
        return valid_u;
    }

    return -1;
}

void prepare_candidates(vector<Server> &servers, vector<Job> &jobs) {
    for (auto &job : jobs) {
        job.candidates.clear();

        for (int i = 0; i < (int)servers.size(); ++i) {
            const Server &server = servers[i];
            int valid_u = get_min_valid_u(server, job);

            if (valid_u == -1) {
                continue;
            }

            int memory_waste = valid_u * server.vg_single - job.v_req;
            int server_gpu_memory = server.g_total * server.vg_single;
            int cpu_waste = server.c_total - job.c_req;
            int mem_waste = server.r_total - job.m_req;

            job.candidates.push_back({
                i,
                valid_u,
                memory_waste,
                server_gpu_memory,
                cpu_waste,
                mem_waste
            });
        }

        sort(job.candidates.begin(), job.candidates.end(), [&](const Candidate &a, const Candidate &b) {
            const Server &sa = servers[a.server_idx];
            const Server &sb = servers[b.server_idx];

            return tie(a.memory_waste,
                       a.valid_u,
                       a.cpu_waste,
                       a.mem_waste,
                       a.server_gpu_memory,
                       sa.sid)
                 < tie(b.memory_waste,
                       b.valid_u,
                       b.cpu_waste,
                       b.mem_waste,
                       b.server_gpu_memory,
                       sb.sid);
        });

        job.compat_cnt = (int)job.candidates.size();
    }
}

long long weight_density(const Job &job) {
    return 1LL * job.weight * 1000000LL / job.proc_len;
}

struct JobOrderKey {
    int compat_cnt;
    long long neg_density;
    int release_t;
    int neg_weight;
    int neg_proc_len;
    int jid;
};

JobOrderKey make_job_key(const Job &job) {
    return {
        job.compat_cnt,
        -weight_density(job),
        job.release_t,
        -job.weight,
        -job.proc_len,
        job.jid
    };
}

bool operator<(const JobOrderKey &a, const JobOrderKey &b) {
    return tie(a.compat_cnt,
               a.neg_density,
               a.release_t,
               a.neg_weight,
               a.neg_proc_len,
               a.jid)
         < tie(b.compat_cnt,
               b.neg_density,
               b.release_t,
               b.neg_weight,
               b.neg_proc_len,
               b.jid);
}

struct ServerChoiceKey {
    int start_t;
    int memory_waste;
    int valid_u;
    int cpu_waste;
    int mem_waste;
    int server_gpu_memory;
    int sid;
};

bool operator<(const ServerChoiceKey &a, const ServerChoiceKey &b) {
    return tie(a.start_t,
               a.memory_waste,
               a.valid_u,
               a.cpu_waste,
               a.mem_waste,
               a.server_gpu_memory,
               a.sid)
         < tie(b.start_t,
               b.memory_waste,
               b.valid_u,
               b.cpu_waste,
               b.mem_waste,
               b.server_gpu_memory,
               b.sid);
}

void run_scheduler(vector<Server> &servers, vector<Job> &jobs) {
    if (jobs.empty()) {
        return;
    }

    prepare_candidates(servers, jobs);

    vector<int> order(jobs.size());
    iota(order.begin(), order.end(), 0);

    sort(order.begin(), order.end(), [&](int ia, int ib) {
        return make_job_key(jobs[ia]) < make_job_key(jobs[ib]);
    });

    for (int job_idx : order) {
        Job &job = jobs[job_idx];

        bool has_best = false;
        ServerChoiceKey best_key{};
        int best_server_idx = -1;
        int best_u = -1;
        int best_start_t = -1;

        for (const auto &cand : job.candidates) {
            Server &server = servers[cand.server_idx];

            int start_t = server.find_window(
                cand.valid_u,
                job.c_req,
                job.m_req,
                job.release_t,
                job.proc_len
            );

            ServerChoiceKey current_key{
                start_t,
                cand.memory_waste,
                cand.valid_u,
                cand.cpu_waste,
                cand.mem_waste,
                cand.server_gpu_memory,
                server.sid
            };

            if (!has_best || current_key < best_key) {
                has_best = true;
                best_key = current_key;
                best_server_idx = cand.server_idx;
                best_u = cand.valid_u;
                best_start_t = start_t;
            }

            if (start_t == job.release_t) {
                break;
            }
        }

        if (best_server_idx == -1) {
            // According to the problem statement this should not happen.
            best_server_idx = 0;
            best_u = max(1, job.g_req);
            best_start_t = job.release_t;
        }

        Server &best_server = servers[best_server_idx];

        job.sid = best_server.sid;
        job.t_start = best_start_t;
        job.u = best_u;
        job.f_finish = best_start_t + job.proc_len;

        best_server.occupy_resource(
            best_start_t,
            best_start_t + job.proc_len,
            best_u,
            job.c_req,
            job.m_req
        );
    }
}

bool check_all_constraints(const vector<Server> &servers, const vector<Job> &jobs) {
    unordered_map<int, const Server *> server_map;
    for (const auto &s : servers) {
        server_map[s.sid] = &s;
    }

    unordered_set<int> seen;

    struct Event {
        int t;
        int dg;
        int dc;
        int dm;
    };

    vector<vector<Event>> events(servers.size());
    unordered_map<int, int> sid_to_index;
    for (int i = 0; i < (int)servers.size(); ++i) {
        sid_to_index[servers[i].sid] = i;
    }

    for (const auto &job : jobs) {
        if (seen.count(job.jid)) return false;
        seen.insert(job.jid);

        auto it = server_map.find(job.sid);
        if (it == server_map.end()) return false;
        const Server &server = *it->second;

        if (job.t_start < job.release_t) return false;
        if (job.f_finish != job.t_start + job.proc_len) return false;
        if (!(job.g_req <= job.u && job.u <= server.g_total)) return false;
        if (job.v_req > job.u * server.vg_single) return false;
        if (job.c_req > server.c_total || job.m_req > server.r_total) return false;

        int idx = sid_to_index[job.sid];
        events[idx].push_back({job.t_start, job.u, job.c_req, job.m_req});
        events[idx].push_back({job.f_finish, -job.u, -job.c_req, -job.m_req});
    }

    if ((int)seen.size() != (int)jobs.size()) return false;

    for (int i = 0; i < (int)servers.size(); ++i) {
        auto &evs = events[i];
        sort(evs.begin(), evs.end(), [](const Event &a, const Event &b) {
            return a.t < b.t;
        });

        int used_g = 0, used_c = 0, used_m = 0;
        int p = 0;

        while (p < (int)evs.size()) {
            int current_t = evs[p].t;
            while (p < (int)evs.size() && evs[p].t == current_t) {
                used_g += evs[p].dg;
                used_c += evs[p].dc;
                used_m += evs[p].dm;
                ++p;
            }

            if (used_g < 0 || used_c < 0 || used_m < 0) return false;
            if (used_g > servers[i].g_total || used_c > servers[i].c_total || used_m > servers[i].r_total) return false;
        }
    }

    return true;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int server_cnt, job_cnt;
    if (!(cin >> server_cnt >> job_cnt)) {
        return 0;
    }

    vector<Server> servers;
    servers.reserve(server_cnt);

    for (int sid = 1; sid <= server_cnt; ++sid) {
        int g_total, vg_single, c_total, r_total;
        cin >> g_total >> vg_single >> c_total >> r_total;
        servers.emplace_back(sid, g_total, vg_single, c_total, r_total);
    }

    vector<Job> jobs;
    jobs.reserve(job_cnt);

    for (int jid = 1; jid <= job_cnt; ++jid) {
        Job job;
        job.jid = jid;
        cin >> job.release_t
            >> job.proc_len
            >> job.g_req
            >> job.v_req
            >> job.c_req
            >> job.m_req
            >> job.weight;
        jobs.push_back(std::move(job));
    }

    run_scheduler(servers, jobs);

    if (DEBUG_MODE && !check_all_constraints(servers, jobs)) {
        return 1;
    }

    sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) {
        return a.jid < b.jid;
    });

    for (const auto &job : jobs) {
        cout << job.jid << ' '
             << job.sid << ' '
             << job.t_start << ' '
             << job.u << ' '
             << job.f_finish << '\n';
    }

    return 0;
}

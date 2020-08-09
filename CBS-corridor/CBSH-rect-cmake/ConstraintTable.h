#pragma once
#include "Conflict.h"
#include <climits>
#include <unordered_set>
#include <set>
class ConstraintTable
{
public:
	int length_max = INT_MAX;
    int kRobust = 1;
	void clear()
	{
		CT.clear();
		CT.resize(CT_paths.size());
	}

	void reset() {auto map_size = CT_paths.size(); CT_paths.clear(); CT_paths.resize(map_size); }
	void insert(int loc, int t_min, int t_max); // insert a constraint
	bool insert_path(int agent_id, const Path& path);
    void delete_path(int agent_id, const Path& path);
	bool is_constrained(int agent_id, int loc, int t) const;
	// bool is_good_malfunction_location(int loc, int t);
    void get_agents(set<int>& conflicting_agents, int loc) const;
    void get_agents(list< pair<int, int> >& agents, int excluded_agent, const pair<int,int>& loc_time_pair) const;
    void get_agents(set<int>& conflicting_agents, int groupsize, int loc) const;
	void get_conflicting_agents(int agent_id, set<int>& conflicting_agents, int loc, int t) const;

    void init(size_t map_size)
    {
        CT_paths.resize(map_size);
    }
private:
    vector< vector<bool> > CT; // this stores the constraints from CBS
    vector< vector<int> > CT_paths; // this stores the already planned paths, the value is the id of the agent
};


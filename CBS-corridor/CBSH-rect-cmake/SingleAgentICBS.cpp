#include "SingleAgentICBS.h"

#include <iostream>
#include <ctime>

template<class Map>
void SingleAgentICBS<Map>::updatePath(LLNode* goal, std::vector<PathEntry> &path)
{
	path.resize(goal->timestep + 1);
	LLNode* curr = goal;
	num_of_conf = goal->num_internal_conf;
	for(int t = goal->timestep; t >= 0; t--)
	{

		path[t].location = curr->loc;
		path[t].actionToHere = curr->heading;
		path[t].position_fraction = curr->position_fraction;
		path[t].malfunction_left = curr->malfunction_left;
		path[t].next_malfunction = curr->next_malfunction;
		path[t].exit_heading = curr->exit_heading;
		path[t].exit_loc = curr->exit_loc;
		delete path[t].conflist;
		path[t].conflist = curr->conflist;
		if (t == goal->timestep && curr->loc != goal_location) {
			path[t].malfunction = true;
		}

		curr->conflist = NULL;
		curr = curr->parent;
	}
}


// iterate over the constraints ( cons[t] is a list of all constraints for timestep t) and return the latest
// timestep which has a constraint involving the goal location
template<class Map>
int SingleAgentICBS<Map>::extractLastGoalTimestep(int goal_location, const std::vector< std::list< std::pair<int, int> > >* cons) {
	if (cons != NULL) {
		for (int t = static_cast<int>(cons->size()) - 1; t > 0; t--) 
		{
			for (std::list< std::pair<int, int> >::const_iterator it = cons->at(t).begin(); it != cons->at(t).end(); ++it)
			{
				if (std::get<0>(*it) == goal_location && it->second < 0) 
				{
					return (t);
				}
			}
		}
	}
	return -1;
}


// input: curr_id (location at time next_timestep-1) ; next_id (location at time next_timestep); next_timestep
//        cons[timestep] is a list of <loc1,loc2> of (vertex/edge) constraints for that timestep.
//inline bool SingleAgentICBS::isConstrained(int curr_id, int next_id, int next_timestep, const std::vector< std::list< std::pair<int, int> > >* cons)  const
//{
//	if (cons == NULL)
//		return false;
//	// check vertex constraints (being in next_id at next_timestep is disallowed)
//	if (next_timestep < static_cast<int>(cons->size()))
//	{
//		for (std::list< std::pair<int, int> >::const_iterator it = cons->at(next_timestep).begin(); it != cons->at(next_timestep).end(); ++it)
//		{
//			if ((std::get<0>(*it) == next_id && std::get<1>(*it) < 0)//vertex constraint
//				|| (std::get<0>(*it) == curr_id && std::get<1>(*it) == next_id)) // edge constraint
//				return true;
//		}
//	}
//	return false;
//}


template<class Map>
int SingleAgentICBS<Map>::numOfConflictsForStep(int curr_id, int next_id, int next_timestep, const bool* res_table, int max_plan_len) {
	int retVal = 0;
	if (next_timestep >= max_plan_len) {
		// check vertex constraints (being at an agent's goal when he stays there because he is done planning)
		if (res_table[next_id + (max_plan_len - 1)*map_size] == true)
			retVal++;
		// Note -- there cannot be edge conflicts when other agents are done moving
	}
	else {
		// check vertex constraints (being in next_id at next_timestep is disallowed)
		if (res_table[next_id + next_timestep*map_size] == true)
			retVal++;
		// check edge constraints (the move from curr_id to next_id at next_timestep-1 is disallowed)
		// which means that res_table is occupied with another agent for [curr_id,next_timestep] and [next_id,next_timestep-1]
		if (res_table[curr_id + next_timestep*map_size] && res_table[next_id + (next_timestep - 1)*map_size])
			retVal++;
	}
	return retVal;
}

template<class Map>
bool SingleAgentICBS<Map>::validMove(int curr, int next) const
{
	if (next < 0 || next >= map_size)
		return false;
	int curr_x = curr / num_col;
	int curr_y = curr % num_col;
	int next_x = next / num_col;
	int next_y = next % num_col;
	return abs(next_x - curr_x) + abs(next_y - curr_y) < 2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// return true if a path found (and updates vector<int> path) or false if no path exists
template<class Map>
bool SingleAgentICBS<Map>::findPath(std::vector<PathEntry> &path, double f_weight, ConstraintTable& constraint_table,
	ReservationTable* res_table, size_t max_plan_len, double lowerbound, std::clock_t start_clock ,int time_limit)
{
	if (constraint_table.is_constrained(start_location, 0))
		return false;
	if (al->agents[agent_id].malfunction_left > 0) {
		for (int i = 0;i < al->agents[agent_id].malfunction_left; i++) {
			if (constraint_table.is_constrained(start_location, i))
				return false;
		}
		if (al->agents[agent_id].position_fraction + al->agents[agent_id].speed < 0.97) {
			int count = 0;
			for (float i = al->agents[agent_id].position_fraction; i < 0.97; i = i + al->agents[agent_id].speed) {
				if (constraint_table.is_constrained(start_location, al->agents[agent_id].malfunction_left+count))
					return false;
				count++;
			}
		}
	}
	else if (al->agents[agent_id].position_fraction + al->agents[agent_id].speed < 0.97) {
		int count = 0;
		for (float i = al->agents[agent_id].position_fraction; i < 0.97; i = i + al->agents[agent_id].speed) {
			if (constraint_table.is_constrained(start_location, count))
				return false;
			count++;
		}
	}

	//is malfunction agent constrained during malfunction state, if yes, return false
	if (al->agents[agent_id].malfunction_left != 0) {
		for (int i = 0; i <= al->agents[agent_id].malfunction_left; i++) {
			if (constraint_table.is_constrained(start_location, i))
				return false;
		}
	}
	num_expanded = 0;
	num_generated = 0;

	hashtable_t::iterator it;  // will be used for find()


	 // generate start and add it to the OPEN list
	LLNode* start = new LLNode(-1, 0, my_heuristic[start_location].heading[start_heading], NULL, 0, 0, false);
	start->heading = start_heading;
	num_generated++;
	start->open_handle = open_list.push(start);
	start->focal_handle = focal_list.push(start);
	start->in_openlist = true;
	start->time_generated = 0;
	OldConfList* conflicts = res_table->findConflict(agent_id, start->loc, start->loc, -1, kRobust);
	start->conflist = conflicts;
	start->num_internal_conf= conflicts->size();

	start->malfunction_left = al->agents[agent_id].malfunction_left;
	start->next_malfunction = al->agents[agent_id].next_malfuntion;

    
	start->position_fraction = al->agents[agent_id].position_fraction;
	start->exit_heading = al->agents[agent_id].exit_heading;
	if (start->exit_heading >= 0) {
		vector<Transition> temp;
		temp = ml->get_transitions(start_location, start->heading, true);
		if (temp.size() == 1) {
			start->exit_loc = temp.front().first;
			start->exit_heading = temp.front().second;
		}
		else
			start->exit_loc = start_location + ml->moves_offset[start->exit_heading];
	}

	int start_h_val = my_heuristic[start_location].heading[start_heading] / al->agents[agent_id].speed;
	if (start->exit_loc >= 0 && al->agents[agent_id].speed < 1) {
		int h1 = my_heuristic[start_location].heading[start_heading];
		int h2 = my_heuristic[start->exit_loc].get_hval(start->exit_heading);
		start_h_val = h1 / al->agents[agent_id].speed
			- (h2 - h1)*al->agents[agent_id].speed;

	}
	start->h_val = start_h_val;


	allNodes_table[start] = start;
	min_f_val = start->getFVal();

	lowerbound = std::max(lowerbound, (double)constraint_table.length_min);

	lower_bound = std::max(lowerbound, f_weight * min_f_val);

	int time_generated = 0;
	int time_check_count = 0;
	std:clock_t runtime;
	/*for (int h = 0; h < my_heuristic.size();h++) {
		for (int heading = 0; heading<5;heading++)
			std::cout << "(" << h << ": heading:"<<-1 <<": "<< my_heuristic[h].heading[-1] << ")";
	}*/

	while (!focal_list.empty()) 
	{
		if (num_generated / 10000 > time_check_count && time_limit != 0) {
			runtime = std::clock() - start_clock;
			time_check_count = num_generated / 10000;
			if (runtime > time_limit) {
				return false;
			}
		}

		LLNode* curr = focal_list.top(); focal_list.pop();
		open_list.erase(curr->open_handle);
// 		cout <<"f: "<< curr->getFVal() <<" g: "<<curr->g_val<<" h: "<<curr->h_val<< endl;
		curr->in_openlist = false;
		num_expanded++;
		//cout << "focal size " << focal_list.size() << endl;
		//cout << "goal_location: " << goal_location << " curr time: " << curr->timestep << " length_min: " << constraint_table.length_min << endl;
		// check if the popped node is a goal
		if ((curr->loc == goal_location ) /*|| (curr->parent!= NULL && curr->next_malfunction==0 && curr->parent->next_malfunction ==1)*/)
		{
			//bool parentAtGoal = false;
			//LLNode* temp = curr;
			//for (int x = 0; x <= kRobust; x++) {
			//	if (temp->parent == NULL) {
			//		break;
			//	}
			//	if (temp->parent->loc == goal_location) {
			//		parentAtGoal = true;
			//	}
			//	temp = temp->parent;
			//}
			//if (curr->parent == NULL /*|| !parentAtGoal*/)
			//{
				//cout << num_generated << endl;

			updatePath(curr, path);

			releaseClosedListNodes(&allNodes_table);

			open_list.clear();
			focal_list.clear();

			allNodes_table.clear();
			goal_nodes.clear();

			return true;
			//}
		}
		

		vector<Transition> transitions;
		if (curr->malfunction_left > 0) {
            //agent is malfunction. useless in flatland 2.2
			Transition move;
			move.first = curr->loc;
			move.second = 4;
			move.position_fraction = curr->position_fraction;
			move.exit_loc = curr->exit_loc;
			move.exit_heading = curr->exit_heading;
			transitions.push_back(move);
		}
        else if(curr->loc == -1){
            Transition move;
			move.first = -1;
			move.second = 4;
			move.position_fraction = curr->position_fraction;
			move.exit_loc = curr->exit_loc;
			move.exit_heading = curr->exit_heading;
			transitions.push_back(move);
            
            Transition move2;
			move2.first = start_location;
			move2.second = 4;
			move2.position_fraction = curr->position_fraction;
			move2.exit_loc = curr->exit_loc;
			move2.exit_heading = curr->exit_heading;
			transitions.push_back(move2);
            
        }
		else if (curr->position_fraction + al->agents[agent_id].speed >= 0.97) {
			if (curr->position_fraction == 0)
				transitions = ml->get_transitions(curr->loc, curr->heading, false);
			else {
				Transition move;
				move.first = curr->exit_loc;
				move.second = curr->exit_heading;
				move.position_fraction = 0;
				transitions.push_back(move);
			}
		}
		else if (curr->position_fraction == 0) {
			transitions = ml->get_exits(curr->loc, curr->heading, al->agents[agent_id].speed, false);

		}
		else { //<0.97 and po_frac not 0
			

			Transition move2;
			move2.first = curr->loc;
			move2.second = curr->heading;
			move2.position_fraction = curr->position_fraction + al->agents[agent_id].speed;
			move2.exit_loc = curr->exit_loc;
			move2.exit_heading = curr->exit_heading;
			transitions.push_back(move2);


		}
        
//         cout<<"current location" << curr->loc<<endl;
// 		cout << "transitions : " ;
// 		for (int i = 0; i < transitions.size(); i++) {
// 			cout << "(" << transitions[i].first << "," << transitions[i].second << ") ";
// 		}
// 		cout << endl;

		for (const auto move : transitions)
		{
			int next_id = move.first;
			time_generated += 1;
			int next_timestep = curr->timestep + 1;

			int next_malfunction_left = curr->malfunction_left > 0? curr->malfunction_left - 1 : curr->malfunction_left;
			int next_next_malfuntion = (curr->next_malfunction > 0 && curr->malfunction_left == 0) ? curr->next_malfunction -1 : curr->next_malfunction;
			if (curr->parent != NULL && next_next_malfuntion == 0 && curr->next_malfunction == 1) {
				next_malfunction_left = this->max_malfunction;
			}
			/*if (max_plan_len <= curr->timestep)
			{
				if (next_id == curr->loc)
				{
					continue;
				}
			}*/
			//if (next_id == 761) {
// 				cout << next_id << " " << next_timestep << endl;
// 				cout << constraint_table.is_constrained(next_id, next_timestep) << endl;
			//}
			if (!constraint_table.is_constrained(next_id, next_timestep) &&
				!constraint_table.is_constrained(curr->loc * map_size + next_id, next_timestep))
			{
				if ((next_next_malfuntion == 0 && curr->malfunction_left == 1) && !constraint_table.is_good_malfunction_location(next_id,next_timestep))
					continue;//if next location not suitable for malfunction, do not generate new node.
				// compute cost to next_id via curr node
				int next_g_val = curr->g_val + 1;
				int next_heading;

				if (curr->heading == -1) //heading == 4 means no heading info
					next_heading = -1;
				else
					if (move.second == 4) //move == 4 means wait
						next_heading = curr->heading;
					else
						next_heading = move.second;
				float next_position_fraction = move.position_fraction;

				
                int next_h_val;
                if (next_id!=-1)
                    next_h_val = my_heuristic[next_id].get_hval(next_heading)/al->agents[agent_id].speed;
                else
                    next_h_val = curr->h_val;
//                 cout<<"next_id "<< next_id <<" curr heading "<< curr->heading<<" next heading "<<next_heading<<" h: "<<next_h_val<<" next_position_fraction "<< next_position_fraction <<endl;
                    
				if (next_id!=-1 && move.exit_loc >= 0 && al->agents[agent_id].speed<1) {
					int h1 = my_heuristic[next_id].get_hval(next_heading);
					int h2 = my_heuristic[move.exit_loc].get_hval(move.exit_heading);
					next_h_val = h1 / al->agents[agent_id].speed
						- (h2-h1)*al->agents[agent_id].speed;

				}
				//cout << "next_h_val " << next_h_val << endl;
				if (next_g_val + next_h_val > constraint_table.length_max)
					continue;


				OldConfList* conflicts = res_table->findConflict(agent_id, curr->loc, next_id, curr->timestep, kRobust);
				int next_internal_conflicts = curr->num_internal_conf + conflicts->size();

				

				// generate (maybe temporary) node
				LLNode* next = new LLNode(next_id, next_g_val, next_h_val,	curr, next_timestep, next_internal_conflicts, false);
				next->heading = next_heading;
				next->actionToHere = move.second;
				next->time_generated = time_generated;
				next->next_malfunction = next_next_malfuntion;
				next->malfunction_left = next_malfunction_left;
				next->position_fraction = next_position_fraction;
				next->exit_heading = move.exit_heading;
				next->exit_loc = move.exit_loc;
				//std::cout << "current: (" << curr->loc << "," << curr->heading << "," << curr->getFVal() <<","<<curr->g_val<<","<<curr->h_val<<","<<curr->position_fraction << ") "
				//	<< "next: (" << next->loc << "," << next->heading << "," << next->getFVal() << "," << next->g_val << "," << next->h_val <<","<<next->position_fraction<< ")"
				//	<< " goal: "<< goal_location<< std::endl;

				// try to retrieve it from the hash table
				it = allNodes_table.find(next);
				if (it == allNodes_table.end() || (next_id == goal_location && constraint_table.length_min > 0) )
				{

					//cout << "Possible child loc: " << next->loc << " heading: " << next->heading << " f: " << next->getFVal() << " g: " << next->g_val << " h: " << next->h_val<< " num_internal_conf: " << next->num_internal_conf << endl;
					//cout << "h: " << my_heuristic[next_id].get_hval(next_heading) << endl;
					

					next->open_handle = open_list.push(next);
					next->in_openlist = true;
					num_generated++;
					if (next->getFVal() <= lower_bound) {
						//cout << "focal size " << focal_list.size() << endl;
						//cout << "put in focal list" << endl;
						next->focal_handle = focal_list.push(next);
						next->in_focallist = true;
						//cout << "focal size " << focal_list.size() << endl;


					}

					if (it == allNodes_table.end())
						allNodes_table[next] = next;
					else
						goal_nodes.push_back(next);
					next->conflist = conflicts;

				}
				else
				{  // update existing node's if needed (only in the open_list)
					delete(next);  // not needed anymore -- we already generated it before
					LLNode* existing_next = (*it).second;

					if (existing_next->in_openlist == true)
					{  // if its in the open list
						if (existing_next->getFVal() > next_g_val + next_h_val ||
							(existing_next->getFVal() == next_g_val + next_h_val && existing_next->num_internal_conf > next_internal_conflicts))
						{
							// if f-val decreased through this new path (or it remains the same and there's less internal conflicts)
							bool add_to_focal = false;  // check if it was above the focal bound before and now below (thus need to be inserted)
							bool update_in_focal = false;  // check if it was inside the focal and needs to be updated (because f-val changed)
							bool update_open = false;
							if ((next_g_val + next_h_val) <= lower_bound)
							{  // if the new f-val qualify to be in FOCAL
								if (existing_next->getFVal() > lower_bound)
									add_to_focal = true;  // and the previous f-val did not qualify to be in FOCAL then add
								else
									update_in_focal = true;  // and the previous f-val did qualify to be in FOCAL then update
							}
							if (existing_next->getFVal() > next_g_val + next_h_val)
								update_open = true;
							// update existing node
							existing_next->g_val = next_g_val;
							existing_next->h_val = next_h_val;
							existing_next->parent = curr;
							existing_next->num_internal_conf = next_internal_conflicts;
							delete(existing_next->conflist);
							existing_next->conflist = conflicts;
							if (update_open) 
								open_list.increase(existing_next->open_handle);  // increase because f-val improved
							if (add_to_focal) 
								existing_next->focal_handle = focal_list.push(existing_next);
							if (update_in_focal) 
								focal_list.update(existing_next->focal_handle);  // should we do update? yes, because number of conflicts may go up or down
						}				
					}
					else 
					{  // if its in the closed list (reopen)
						if (existing_next->getFVal() > next_g_val + next_h_val ||
							(existing_next->getFVal() == next_g_val + next_h_val && existing_next->num_internal_conf > next_internal_conflicts)) 
						{
							// if f-val decreased through this new path (or it remains the same and there's less internal conflicts)
							existing_next->g_val = next_g_val;
							existing_next->h_val = next_h_val;
							existing_next->parent = curr;
							existing_next->num_internal_conf = next_internal_conflicts;
							existing_next->open_handle = open_list.push(existing_next);
							existing_next->in_openlist = true;
							delete(existing_next->conflist);
							existing_next->conflist = conflicts;
							if (existing_next->getFVal() <= lower_bound)
								existing_next->focal_handle = focal_list.push(existing_next);
						}
					}  // end update a node in closed list
				}  // end update an existing node
			}// end if case forthe move is legal
		}  // end for loop that generates successors
		//cout << "focal list size"<<focal_list.size() << endl;
		// update FOCAL if min f-val increased
		if (open_list.size() == 0)  // in case OPEN is empty, no path found
			break;
		LLNode* open_head = open_list.top();
		

		if (open_head->getFVal() > min_f_val) 
		{

			double new_min_f_val = open_head->getFVal();
			double new_lower_bound = std::max(lowerbound, f_weight * new_min_f_val);

			for (LLNode* n : open_list) 
			{

				if (!n->in_focallist && n->getFVal() > lower_bound && n->getFVal() <= new_lower_bound) {

					n->focal_handle = focal_list.push(n);
					n->in_focallist = true;
				}
			}

			min_f_val = new_min_f_val;
			lower_bound = new_lower_bound;

		}

	}  // end while loop

	  // no path found
	releaseClosedListNodes(&allNodes_table);
	open_list.clear();
	focal_list.clear();
	allNodes_table.clear();
	goal_nodes.clear();

	return false;
}

template<class Map>
inline void SingleAgentICBS<Map>::releaseClosedListNodes(hashtable_t* allNodes_table)
{

	hashtable_t::iterator it;
	for (it = allNodes_table->begin(); it != allNodes_table->end(); ++it) {

			delete ((*it).second);
	}
	for (auto node : goal_nodes)
		delete node;

}

template<class Map>
SingleAgentICBS<Map>::SingleAgentICBS(int start_location, int goal_location,  Map* ml1, AgentsLoader* al,int agent_id, int start_heading, int kRobust):ml(ml1)
{
	this->al = al;
	this->agent_id = agent_id;
	this->start_heading = start_heading;

	this->start_location = start_location;
	this->goal_location = goal_location;
	

	this->map_size = ml->cols*ml->rows;

	this->num_expanded = 0;
	this->num_generated = 0;

	this->lower_bound = 0;
	this->min_f_val = 0;

	this->num_col = ml->cols;

	this->kRobust = kRobust;
	// initialize allNodes_table (hash table)
	empty_node = new LLNode();
	empty_node->loc = -1;
	deleted_node = new LLNode();
	deleted_node->loc = -2;
	allNodes_table.set_empty_key(empty_node);
	allNodes_table.set_deleted_key(deleted_node);

}

template<class Map>
SingleAgentICBS<Map>::~SingleAgentICBS()
{
	delete (empty_node);
	delete (deleted_node);
}

template class SingleAgentICBS<MapLoader>;
template class SingleAgentICBS<FlatlandLoader>;
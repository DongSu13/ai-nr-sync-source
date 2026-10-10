function branch=nr_v4_branch(context,public,truth,budget,maxSearches,deps)
% Offline paired teacher rollout. Truth used only AFTER both policies finish.
if nargin<6,deps=struct('search',@nr_v4_search,'decode',@nr_v4_decode,'reconstruct',@nr_v4_reconstruct);end
initial=context.initial;start=initial.s.decode_count;limit=min(budget,start+2);
assert(start<limit && initial.s.search_count<maxSearches);
teacher=struct('rank','dmrs_fit','gate','energy_1','mode','sic');
skip=nr_v4_control(initial.s.rx,public,teacher,limit,maxSearches,deps,initial);
cancelInitial=initial;cancelInitial.s.rx=initial.s.rx-context.estimate;
[cancelInitial.c,~]=deps.search(cancelInitial.s.rx,public);
cancelInitial.s.search_count=initial.s.search_count+1;cancelInitial.tried=[];
cancel=nr_v4_control(cancelInitial.s.rx,public,teacher,limit,maxSearches,deps,cancelInitial);
old=nr_v4_evaluate(struct('outputs',initial.s.outputs),truth);
ms=nr_v4_evaluate(skip,truth);mc=nr_v4_evaluate(cancel,truth);
branch=struct('features',context.features,'benefit',mc.recovered_users-ms.recovered_users,...
    'skip_new_users',ms.recovered_users-old.recovered_users,...
    'cancel_new_users',mc.recovered_users-old.recovered_users,...
    'skip_decode_calls',skip.decode_count-start,'cancel_decode_calls',cancel.decode_count-start,...
    'start_decode_count',start,'start_search_count',initial.s.search_count,...
    'skip',skip,'cancel',cancel);
end

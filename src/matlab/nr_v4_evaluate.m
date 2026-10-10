function metrics=nr_v4_evaluate(run,truth)
% Offline only: policy never receives this evaluator or its truth argument.
users=truth.users;outputs=run.outputs;recovered=[];incorrect=0;
if ~isempty(outputs),assert(numel(unique([outputs.pci]))==numel(outputs),'Duplicate output PCI');end
for j=1:numel(outputs)
    match=[];if ~isempty(users),match=find([users.pci]==outputs(j).pci);end
    if ~isempty(match) && isequal(outputs(j).payload,users(match).payload)
        recovered(end+1)=match;
    else,incorrect=incorrect+1;end
end
weak=[];if ~isempty(users),weak=find([users.power_db]==min([users.power_db]));end
metrics=struct('recovered_users',numel(recovered),'transmitted_users',numel(users),...
    'weak_recovered',numel(intersect(weak,recovered)),'weak_transmitted',numel(weak),...
    'incorrect_outputs',incorrect,'all_recovered',~isempty(users) && numel(recovered)==numel(users));
end

function sample=nr_v4_collect(record)
% Training-only offline labeling. The online policy sees neither truth nor
% outcomes of counterfactual branches. Validation/test must not call this.
assert(ismember(record.split,{'train','development','unit'}),'Offline labels forbidden on holdouts');
t=tic;[rx,truth,public]=nr_v4_waveform(record);rng(record.seed+123000000,'twister');
policy=struct('rank','dmrs_fit','gate','energy_1','mode','sic','capture',true);
if mod(record.seed,2)==0,policy.rank='random';policy.gate='random';end
run=nr_v4_control(rx,public,policy,6,4);cache=cell(4,12);
x=zeros(0,17);y=zeros(0,1);rankRecords=struct([]);offlineCalls=0;
for j=1:numel(run.rank_contexts)
    context=run.rank_contexts{j};s=context.s;
    for k=1:numel(context.c)
        c=context.c(k);
        if ismember(c.pci,context.tried) || (~isempty(s.outputs) && ismember(c.pci,[s.outputs.pci])),continue;end
        if isempty(cache{s.search_count,k})
            cache{s.search_count,k}=nr_v4_decode(c);offlineCalls=offlineCalls+1;
        end
        d=cache{s.search_count,k};match=[];
        if ~isempty(truth.users),match=find([truth.users.pci]==c.pci);end
        label=d.crc && ~isempty(match) && isequal(d.payload,truth.users(match).payload);
        x(end+1,:)=nr_v4_features(c,s);y(end+1,1)=label;
        row=struct('round',s.search_count,'used',s.decode_count,'candidate_id',c.id,...
            'pci',c.pci,'crc',d.crc,'payload',d.payload,'label',label);
        if isempty(rankRecords),rankRecords=row;else,rankRecords(end+1)=row;end
    end
end
branches=struct([]);
for j=1:numel(run.decision_contexts)
    b=nr_v4_branch(run.decision_contexts{j},public,truth,6,4);
    if isempty(branches),branches=b;else,branches(end+1)=b;end
end
run=rmfield(run,{'rank_contexts','decision_contexts'});
sample=record;sample.truth=truth;sample.online_run=run;
sample.rank_x=x;sample.rank_y=y;sample.rank_records=rankRecords;sample.branches=branches;
sample.offline_label_decode_calls=offlineCalls;sample.elapsed_seconds=toc(t);
end

function run_nr_v4(split,first,last,mode,policyFile)
% One sequential batch. Caller owns a single-process lock and verifies source
% hashes before/after execution. Existing scene data are never overwritten.
root=fullfile('outputs','nr_gain_v4');records=jsondecode(fileread(fullfile(root,'manifests',[split '.json'])));
assert(first>=1 && last<=numel(records) && first<=last);
assert(ismember(mode,{'collect','evaluate'}));
if strcmp(mode,'collect')
    assert(ismember(split,{'train','development'}));folder=fullfile(root,'collections',split);job=[];
else
    job=jsondecode(fileread(policyFile));assert(~isempty(regexp(job.name,'^[A-Za-z0-9_]+$','once')));
    folder=fullfile(root,'campaigns',job.name,split);
    if ismember(split,{'test','stress','noise'})
        assert(isfile(fullfile(root,'freeze.json')),'Holdout must be frozen');
    end
end
if ~isfolder(folder),mkdir(folder);end
for index=first:last
    record=records(index);path=fullfile(folder,sprintf('%d.json',record.seed));
    if isfile(path)
        old=jsondecode(fileread(path));assert(strcmp(old.receiver_hash,record.receiver_hash) && strcmp(old.config_hash,record.config_hash),'Resume hash mismatch');
        if strcmp(mode,'evaluate'),assert(strcmp(old.protocol_id,job.protocol_id));end
        continue;
    end
    timer=tic;
    if strcmp(mode,'collect')
        result=nr_v4_collect(record);
        fprintf('COLLECT %s %d/%d rows=%d branches=%d %.2fs\n',split,index,last,size(result.rank_x,1),numel(result.branches),toc(timer));
    else
        [rx,truth,public]=nr_v4_waveform(record);result=record;result.truth=truth;
        result.protocol_id=job.protocol_id;result.runs=struct([]);
        for p=1:numel(job.policies)
            if iscell(job.policies),entry=job.policies{p};else,entry=job.policies(p);end
            useBudgets=job.budgets;
            if isfield(entry,'budgets'),useBudgets=entry.budgets;end
            assert(all(ismember(useBudgets,job.budgets)));
            for budget=useBudgets(:)'
                online=nr_v4_control(rx,public,entry.policy,budget,4);
                % Exported models are stored once in the immutable job file.
                for field={'rank_model','gate_model'}
                    if isfield(online.policy,field{1}),online.policy=rmfield(online.policy,field{1});end
                end
                row=struct('name',entry.name,'budget',budget,'run',online,'metrics',nr_v4_evaluate(online,truth));
                if isempty(result.runs),result.runs=row;else,result.runs(end+1)=row;end
            end
        end
        result.elapsed_seconds=toc(timer);
        fprintf('EVALUATE %s %s %d/%d methods=%d %.2fs\n',job.name,split,index,last,numel(result.runs),result.elapsed_seconds);
    end
    fid=fopen([path '.tmp'],'w','n','UTF-8');assert(fid~=-1);
    fprintf(fid,'%s',jsonencode(result));fclose(fid);movefile([path '.tmp'],path);
end
end

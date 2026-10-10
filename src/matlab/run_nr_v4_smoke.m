function run_nr_v4_smoke
% Serial engineering diagnostics. These scenes never enter AI performance claims.
root=fullfile('outputs','nr_gain_v4');records=jsondecode(fileread(fullfile(root,'manifests','smoke.json')));
folder=fullfile(root,'smoke');if ~isfolder(folder),mkdir(folder);end
for j=1:numel(records)
    record=records(j);path=fullfile(folder,sprintf('%d.json',record.seed));
    if isfile(path)
        old=jsondecode(fileread(path));assert(strcmp(old.receiver_hash,record.receiver_hash) && strcmp(old.config_hash,record.config_hash));
        continue;
    end
    t=tic;[rx,truth,public,diagnostic]=nr_v4_waveform(record);waveTime=toc(t);
    policy=struct('rank','dmrs_fit','gate','always','mode','sic');
    runAlways=nr_v4_control(rx,public,policy,6,4);
    policy.gate='energy_1';runGuard=nr_v4_control(rx,public,policy,6,4);
    policy.mode='no_sic';runNone=nr_v4_control(rx,public,policy,6,4);
    result=record;result.truth=truth;
    result.runs=struct('always',runAlways,'guard',runGuard,'no_sic',runNone);
    result.metrics=struct('always',nr_v4_evaluate(runAlways,truth),...
        'guard',nr_v4_evaluate(runGuard,truth),'no_sic',nr_v4_evaluate(runNone,truth));
    result.diagnostic=struct('first_cancel_component_relative_error',NaN,'other_users_relative_distortion',NaN);
    if ~isempty(runAlways.events)
        ci=find([runAlways.events.cancelled],1);
        if ~isempty(ci)
            event=runAlways.events(ci);match=find([truth.users.pci]==event.pci);
            if ~isempty(match) && isequal(event.payload,truth.users(match).payload)
                c=nr_v4_search(rx,public);candidate=c(event.candidate_id);
                decoded=struct('crc',true,'pci',event.pci,'payload',event.payload);
                estimate=nr_v4_reconstruct(candidate,decoded,rx,public);
                target=diagnostic.components(:,match);error=estimate-target;
                result.diagnostic.first_cancel_component_relative_error=norm(error)/max(norm(target),eps);
                others=sum(diagnostic.components,2)-target;
                if norm(others)>0,result.diagnostic.other_users_relative_distortion=norm(error)/norm(others);end
            end
        end
    end
    result.waveform_seconds=waveTime;result.elapsed_seconds=toc(t);
    fid=fopen([path '.tmp'],'w','n','UTF-8');assert(fid~=-1);
    fprintf(fid,'%s',jsonencode(result));fclose(fid);movefile([path '.tmp'],path);
    fprintf('SMOKE %d/%d %s users=%d noSIC=%d always=%d guard=%d %.2fs\n',j,numel(records),...
        record.cell,record.config.nu,result.metrics.no_sic.recovered_users,...
        result.metrics.always.recovered_users,result.metrics.guard.recovered_users,result.elapsed_seconds);
end
fprintf('NR V4 SMOKE COMPLETE\n');
end

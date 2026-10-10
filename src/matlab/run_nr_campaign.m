function run_nr_campaign(split,first,last)
% Idempotent atomic per-scene output; explicit split seed ranges.
base=struct('train',410000,'validation',510000,'development',610000,'test',710000);
assert(isfield(base,split));out=fullfile('outputs','nr_gain_v2',split);if ~isfolder(out),mkdir(out);end
for index=first:last
    path=fullfile(out,sprintf('%06d.json',index));if isfile(path),continue;end
    tic;s=nr_candidate_scene(base.(split)+index);s.split=split;s.elapsed_seconds=toc;
    fid=fopen([path '.tmp'],'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(s));fclose(fid);
    movefile([path '.tmp'],path);fprintf('%s %d/%d %.2fs labels=%d/%d\n',split,index,last,s.elapsed_seconds,sum([s.candidates.label]),numel(s.candidates));
end
end

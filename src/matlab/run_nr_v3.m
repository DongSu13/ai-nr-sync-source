function run_nr_v3(split,first,last)
root=fullfile('outputs','nr_gain_v3');records=jsondecode(fileread(fullfile(root,'manifests',[split '.json'])));
assert(first>=1 && last<=numel(records));sha=nr_v3_receiver_hash;
for index=first:last
    record=records(index);assert(strcmp(record.receiver_hash,sha),'Receiver hash mismatch');
    folder=fullfile(root,'scenes',split,record.cell);if ~isfolder(folder),mkdir(folder);end
    path=fullfile(folder,sprintf('%d.json',record.seed));
    if isfile(path)
        old=jsondecode(fileread(path));assert(strcmp(old.config_hash,record.config_hash),'Config conflict');
        assert(strcmp(old.receiver_hash,sha),'Existing scene uses other receiver');continue;
    end
    t=tic;s=nr_v3_scene(record);s.elapsed_seconds=toc(t);
    fid=fopen([path '.tmp'],'w','n','UTF-8');assert(fid~=-1);fprintf(fid,'%s',jsonencode(s));fclose(fid);
    movefile([path '.tmp'],path);fprintf('%s %d/%d seed=%d %.2fs positives=%d/12\n',split,index,last,record.seed,s.elapsed_seconds,sum([s.candidates.label]));
end
end

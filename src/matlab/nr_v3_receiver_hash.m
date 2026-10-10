function hex=nr_v3_receiver_hash
names={'nr_v3_grid.m','nr_v3_search.m','nr_v3_scene.m','run_nr_v3.m','nr_v3_receiver_hash.m'};
md=java.security.MessageDigest.getInstance('SHA-256');root=fileparts(mfilename('fullpath'));
for i=1:numel(names)
    fid=fopen(fullfile(root,names{i}),'rb');assert(fid~=-1);bytes=fread(fid,Inf,'*uint8');fclose(fid);
    md.update(uint8([names{i} char(0)]));md.update(bytes);md.update(uint8(0));
end
hex=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

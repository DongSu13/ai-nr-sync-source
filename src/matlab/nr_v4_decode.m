function decoded=nr_v4_decode(candidate)
% The sole v4 Polar call: controller must debit one attempt before entry.
t=tic;[~,err,payload]=nrBCHDecode(candidate.llr,8,4,candidate.pci);
decoded=struct('pci',candidate.pci,'crc',err==0,'payload',double(payload(:))',...
    'seconds',toc(t));
end

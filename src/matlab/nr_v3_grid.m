function grid=nr_v3_grid(pci,payload)
ssb=zeros(240,4);ssb(nrPSSIndices)=nrPSS(mod(pci,3));ssb(nrSSSIndices)=nrSSS(pci);
ssb(nrPBCHIndices(pci))=nrPBCH(nrBCH(double(payload(:)),0,0,4,0,pci),pci,0);
ssb(nrPBCHDMRSIndices(pci))=nrPBCHDMRS(pci,0);
grid=zeros(240,14);grid(:,3:6)=ssb;
end

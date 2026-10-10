function test_nr_v4_receiver
% Real numerical tests: hidden decode, wrong CFO sign, misplaced CP/grid,
% truth-dependent waveform generator and CRC-free cancellation must fail.
assert(exist('nr_v4_search','file')==2,'v4 online search not implemented');
carrier=nrCarrierConfig('NSizeGrid',20,'SubcarrierSpacing',15);
public=struct('carrier',carrier,'ofdm',nrOFDMInfo(carrier));
pci=123;payload=mod((1:24)',2);w=nrOFDMModulate(carrier,nr_v3_grid(pci,payload),'Windowing',0);
for offset=[0 900]
    for cfo=[-1500 1500]
        rx=zeros(numel(w)+4096,1);rx(offset+(1:numel(w)))=w;
        rx=rx.*exp(1j*2*pi*cfo*(0:numel(rx)-1)'/public.ofdm.SampleRate);
        c=nr_v4_search(rx,public);assert(numel(c)==12);
        assert(~isfield(c,'crc') && ~isfield(c,'payload'),'Search leaked decode results');
        ix=find([c.pci]==pci);assert(~isempty(ix));ok=false;
        for k=ix
            d=nr_v4_decode(c(k));
            if d.crc && isequal(d.payload,payload')
                estimate=nr_v4_reconstruct(c(k),d,rx,public);
                relative=norm(estimate-rx)/norm(rx);
                fprintf('AWGN offset=%d CFO=%d reconstruction=%.6g\n',offset,cfo,relative);
                assert(relative<.01,'Noiseless reconstruction inaccurate');ok=true;break;
            end
        end
        assert(ok,'Boundary payload recovery failed');
        rejected=false;bad=struct('crc',false,'payload',payload');
        try,nr_v4_reconstruct(c(ix(1)),bad,rx,public);catch,rejected=true;end
        assert(rejected,'CRC-failed reconstruction allowed');
    end
end
cfg=struct('nu',1,'snr_db',100,'profile','TDL-C','spread_s',300e-9,...
    'doppler',0,'cfo_max',0,'gap_db',6,'timing_mode','random','pss_mode','random','noise_scale',1);
r=struct('seed',49090001,'config',cfg);
for profile={'AWGN','TDL-A','TDL-C'}
    r.config.profile=profile{1};[rx,truth,pub,diag]=nr_v4_waveform(r);
    assert(~isfield(pub,'users') && ~isfield(pub,'noise_variance'));
    c=nr_v4_search(rx,pub);ix=find([c.pci]==truth.users(1).pci);ok=false;
    for k=ix
        d=nr_v4_decode(c(k));
        if d.crc && isequal(d.payload,truth.users(1).payload)
            estimate=nr_v4_reconstruct(c(k),d,rx,pub);
            relative=norm(estimate-diag.components(:,1))/norm(diag.components(:,1));
            fprintf('%s multipath reconstruction=%.6g\n',profile{1},relative);
            assert(relative<1,'Reconstruction worse than no cancellation');ok=true;break;
        end
    end
    assert(ok,'High-SNR single-user decode failed');
end
r.config.nu=0;r.config.noise_reference=.01;
[rx,truth,pub]=nr_v4_waveform(r);c=nr_v4_search(rx,pub);
assert(isempty(truth.users) && mean(abs(rx).^2)>0);
assert(all(arrayfun(@(x)all(isfinite(struct2array(x.features))),c)));
r.config.nu=3;r.config.pss_mode='same';r.config.profile='TDL-C';
[~,truth]=nr_v4_waveform(r);
assert(numel(unique([truth.users.channel_seed]))==3);
assert(numel(unique([truth.users.pci]))==3 && numel(unique(mod([truth.users.pci],3)))==1);
fprintf('NR V4 RECEIVER TESTS PASS\n');
end

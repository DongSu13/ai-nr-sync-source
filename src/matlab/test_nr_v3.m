function test_nr_v3
assert(exist('nr_v3_scene','file')==2,'v3 configurable receiver is not implemented');
cfg=struct('nu',1,'snr_db',100,'profile','AWGN','spread_s',300e-9,...
    'doppler',30,'cfo_max',0,'gap_db',6,'timing_mode','random','pss_mode','random','noise_scale',1);
r=struct('seed',19090001,'split','smoke','suite','unit','cell','single','config',cfg,'config_hash','unit');
s=nr_v3_scene(r);assert(numel(s.candidates)==12);assert(any([s.candidates.label]));
assert(s.noise_variance>0);assert(all(isfinite(struct2array(s.candidates(1).features))));
for k=1:numel(s.candidates)
    assert(s.candidates(k).features.dmrs_power>0);
end
r.config.nu=0;r.config.noise_reference=0.01;r.config.noise_scale=4;
s=nr_v3_scene(r);assert(isempty(s.users));assert(~any([s.candidates.label]));
assert(abs(s.noise_variance-0.04)<1e-12);assert(s.received_power>0);
r.config.nu=6;r.config.profile='TDL-D';r.config.snr_db=0;r.config.cfo_max=1500;
r.config.spread_s=1000e-9;r.config.pss_mode='same';r.config.timing_mode='near';
s=nr_v3_scene(r);assert(numel(s.users)==6);assert(numel(unique([s.users.channel_seed]))==6);
assert(numel(unique(mod([s.users.pci],3)))==1);assert(range([s.users.offset])<=8);
assert(all(isfinite([s.candidates.cfo])));assert(numel(s.candidates)==12);
carrier=nrCarrierConfig('NSizeGrid',20,'SubcarrierSpacing',15);ofdm=nrOFDMInfo(carrier);
for profile={'TDL-A','TDL-C','TDL-D'}
    r.config.nu=1;r.config.profile=profile{1};r.config.snr_db=100;r.config.cfo_max=0;
    r.config.spread_s=300e-9;r.config.doppler=0;r.config.pss_mode='random';
    s=nr_v3_scene(r);u=s.users(1);
    assert(u.power_window_start==u.offset+u.filter_delay+sum(ofdm.SymbolLengths(1:2))+1);
    assert(u.power_window_length==sum(ofdm.SymbolLengths(3:6)));
    assert(abs(s.noise_variance*1e10-s.ref_power)<1e-12);
    assert(any([s.candidates.label]));
end
% Exact no-noise search boundaries, with no true offset or CFO passed to search.
payload=mod((1:24)',2);pci=123;w=nrOFDMModulate(carrier,nr_v3_grid(pci,payload),'Windowing',0);
for offset=[0 900]
    for cfo=[-1500 1500]
        rx=zeros(numel(w)+4096,1);rx(offset+(1:numel(w)))=w;
        rx=rx.*exp(1j*2*pi*cfo*(0:numel(rx)-1)'/ofdm.SampleRate);
        c=nr_v3_search(rx,carrier,ofdm);
        recovered=arrayfun(@(z)z.pci==pci && z.crc && isequal(z.payload,payload'),c);
        assert(any(recovered),'Noiseless boundary payload recovery failed');
    end
end
fprintf('NR V3 MATLAB TESTS PASS\n');
end

function test_nr_v4_collect
assert(exist('nr_v4_collect','file')==2,'training collector missing');
cfg=struct('nu',2,'snr_db',6,'profile','AWGN','spread_s',300e-9,'doppler',0,...
    'cfo_max',500,'gap_db',12,'timing_mode','random','pss_mode','different','noise_scale',1);
r=struct('seed',49080001,'config',cfg,'split','unit');
a=nr_v4_collect(r);b=nr_v4_collect(r);
assert(isequal(a.rank_x,b.rank_x) && isequal(a.rank_y,b.rank_y));
assert(size(a.rank_x,2)==17 && size(a.rank_x,1)==numel(a.rank_y));
assert(any(a.rank_y) && any(~a.rank_y));
assert(numel(a.branches)<=4 && ~isempty(a.branches));
for j=1:numel(a.branches)
    branch=a.branches(j);
    assert(numel(branch.features)==21);
    assert(branch.benefit==branch.cancel_new_users-branch.skip_new_users);
    assert(branch.skip_decode_calls<=2 && branch.cancel_decode_calls<=2);
end
r.config.nu=0;r.config.noise_reference=.01;
a=nr_v4_collect(r);assert(~any(a.rank_y) && isempty(a.branches));
assert(size(a.rank_x,1)>0 && a.online_run.decode_count<=6);
assert(exist('run_nr_v4','file')==2,'atomic batch runner missing');
fprintf('NR V4 COLLECTOR TESTS PASS\n');
end

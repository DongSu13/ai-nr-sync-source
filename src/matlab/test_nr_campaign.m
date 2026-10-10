function test_nr_campaign
% Fails if the search leaks transmitter identity or skips measurements of negatives.
s=nr_candidate_scene(123, true);
c=s.candidates;
assert(any([c.label]), 'Noiseless receiver must recover the known payload.');
assert(any(~[c.label]), 'Search must retain naturally occurring incorrect hypotheses.');
for k=1:numel(c)
    v=struct2array(c(k).features);
    assert(all(isfinite(v)), 'All hypotheses require finite measured features.');
    assert(c(k).features.dmrs_power>0, 'Negative features must not be fabricated zeros.');
end
assert(numel(unique([s.users.channel_seed]))==numel(s.users));
fprintf('NR campaign smoke PASS: %d candidates, %d confirmed\n',numel(c),sum([c.label]));
end

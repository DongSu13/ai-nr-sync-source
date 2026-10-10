function test_nr_v4_predict
assert(exist('nr_v4_predict','file')==2,'MATLAB predictor missing');
fixtures=jsondecode(fileread(fullfile('outputs','nr_gain_v4','predictor_parity.json')));
if ~iscell(fixtures),fixtures=num2cell(fixtures);end
for j=1:numel(fixtures)
    f=fixtures{j};actual=nr_v4_predict(f.artifact,f.x);
    assert(max(abs(actual(:)-f.expected(:)))<1e-10,'Python/MATLAB model prediction mismatch');
end
fprintf('NR V4 PREDICTOR PARITY PASS %d models\n',numel(fixtures));
end

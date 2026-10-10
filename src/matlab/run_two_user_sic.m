function result = run_two_user_sic(configPath, outputPath)
%RUN_TWO_USER_SIC Two-user NR SSB acquisition with PBCH-aided SIC.
% Each transmitted SSB uses the 3GPP PSS, SSS, PBCH, PBCH DM-RS and BCH
% Polar processing supplied by 5G Toolbox. The receiver has no access to
% the transmitted PCI or BCH payload while detecting either user.
if nargin < 1
    configPath = fullfile('configs', 'stage2_two_user.json');
end
if nargin < 2
    outputPath = fullfile('outputs', 'stage2_two_user_sic.json');
end
if isstruct(configPath)
    cfg = configPath;
else
    cfg = jsondecode(fileread(configPath));
end
rng(cfg.seed, 'twister');
carrier = nrCarrierConfig;
carrier.NSizeGrid = cfg.nsize_grid;
carrier.SubcarrierSpacing = cfg.scs_khz;
[referenceWaveform, ~] = localBuildSSB(carrier, cfg.users(1).pci, zeros(24, 1));
totalLength = max([cfg.users.timing_offset_samples]) + numel(referenceWaveform) + 128;
clean = zeros(totalLength, 1);
for userIndex = 1:numel(cfg.users)
    user = cfg.users(userIndex);
    transportBlock = randi([0 1], 24, 1);
    [waveform, ~] = localBuildSSB(carrier, user.pci, transportBlock);
    amplitude = 10^(user.power_db / 20);
    clean = clean + localPlaceWaveform(waveform, user.timing_offset_samples, ...
        user.cfo_hz, amplitude, totalLength, localSampleRate(carrier));
end
channelName = 'AWGN';
if isfield(cfg, 'channel')
    channelName = cfg.channel;
end
if strcmp(channelName, 'TDL-C')
    clean = localApplyTDLC(clean, localSampleRate(carrier), cfg.seed);
end
noiseVariance = mean(abs(clean).^2) / (10^(cfg.snr_db / 10));
received = clean + sqrt(noiseVariance / 2) * (randn(size(clean)) + 1j * randn(size(clean)));
searchLimit = min(totalLength - numel(referenceWaveform), max([cfg.users.timing_offset_samples]) + 1024);

% No-SIC reference: only the strongest single-pass hypothesis is retained.
[noSICCandidate, noSICDetails] = localDetect(received, carrier, cfg.cfo_search_hz, ...
    searchLimit, noiseVariance);

% SIC step 1: decode and reconstruct the strongest candidate from its own
% recovered BCH block and DM-RS channel estimate, then remove it.
strongCandidate = noSICCandidate;
residual = received;
if noSICDetails.pbch_crc_ok
    reconstructed = localBuildSSB(carrier, noSICCandidate.pci, noSICDetails.transport_block);
    residual = received - localPlaceWaveform(reconstructed, noSICCandidate.timing_offset_samples, ...
        noSICCandidate.cfo_hz, noSICDetails.channel_gain, totalLength, localSampleRate(carrier));
end

% SIC step 2: rerun the same blind detector on the residual waveform.
[weakCandidate, weakDetails] = localDetect(residual, carrier, cfg.cfo_search_hz, ...
    searchLimit, noiseVariance);
truth = struct('users', cfg.users);
noSIC = struct('detected_pci', noSICCandidate.pci, ...
    'pbch_crc_ok', logical(noSICCandidate.pbch_crc_ok), ...
    'correlation_metric', noSICCandidate.correlation_metric);
metrics = struct( ...
    'pre_sic_power', mean(abs(received).^2), ...
    'post_sic_residual_power', mean(abs(residual).^2), ...
    'strong_user_pci_match', logical(strongCandidate.pci == cfg.users(1).pci), ...
    'sic_recovered_weaker_user', logical(weakCandidate.pci == cfg.users(2).pci && weakDetails.pbch_crc_ok), ...
    'noise_variance', noiseVariance);
status = 'failure';
failureReason = 'SIC did not confirm both configured users.';
if strongCandidate.pbch_crc_ok && weakCandidate.pbch_crc_ok && ...
        metrics.strong_user_pci_match && metrics.sic_recovered_weaker_user && ...
        metrics.post_sic_residual_power < metrics.pre_sic_power
    status = 'success';
    failureReason = '';
end
result = struct('stage', 'stage2', 'status', status, 'failure_reason', failureReason, ...
    'config', cfg, 'truth', truth, 'no_sic', noSIC, ...
    'candidates', [strongCandidate weakCandidate], 'metrics', metrics);
[result.ai_records, result.ai_candidates] = localBuildAIRecords(cfg, received, carrier, strongCandidate, ...
    weakCandidate, metrics, noSICDetails);
if ~isempty(outputPath)
    localWriteJSON(result, outputPath);
end

function waveform = localApplyTDLC(waveform, sampleRate, seed)
channel = nrTDLChannel;
channel.DelayProfile = 'TDL-C';
channel.DelaySpread = 300e-9;
channel.MaximumDopplerShift = 30;
channel.SampleRate = sampleRate;
channel.NumReceiveAntennas = 1;
channel.Seed = mod(seed, 2^32 - 1);
waveform = channel(waveform);
end
end

function [records, candidates] = localBuildAIRecords(cfg, received, carrier, strongCandidate, weakCandidate, metrics, noSICDetails)
strongLabel = localIsConfirmedTruth(strongCandidate, cfg.users);
weakLabel = localIsConfirmedTruth(weakCandidate, cfg.users);
strongFeatures = localFeatureStruct(strongCandidate, metrics.pre_sic_power, 0);
weakFeatures = localFeatureStruct(weakCandidate, metrics.pre_sic_power, metrics.post_sic_residual_power);

falsePCI = mod(strongCandidate.pci + 137 + mod(cfg.seed, 97), 1008);
while any(falsePCI == [cfg.users.pci])
    falsePCI = mod(falsePCI + 1, 1008);
end
wrongCandidate = localProbeWrongPCI(received, carrier, strongCandidate, falsePCI);
wrongFeatures = localFeatureStruct(wrongCandidate, metrics.pre_sic_power, 0);
records = [ ...
    localAIRecord(cfg.seed, 'no_sic', strongFeatures, strongLabel), ...
    localAIRecord(cfg.seed, 'sic_residual', weakFeatures, weakLabel), ...
    localAIRecord(cfg.seed, 'wrong_pci', wrongFeatures, false)];
candidates = [strongCandidate weakCandidate wrongCandidate];

% The decoded block is intentionally only used by SIC reconstruction above.
% Keeping this reference prevents a future refactor from adding it as a feature.
assert(isfield(noSICDetails, 'pbch_crc_ok'));
end

function isTruth = localIsConfirmedTruth(candidate, users)
isTruth = candidate.pbch_crc_ok && any(candidate.pci == [users.pci]);
end

function features = localFeatureStruct(candidate, receivedPower, residualPower)
features = struct('pss_metric', candidate.pss_metric, ...
    'sss_metric', candidate.sss_metric, ...
    'correlation_metric', candidate.correlation_metric, ...
    'abs_cfo_hz', abs(candidate.cfo_hz), ...
    'received_power', receivedPower, ...
    'residual_power', residualPower);
end

function record = localAIRecord(seed, source, features, label)
record = struct('split_seed', seed, 'candidate_source', source, ...
    'features', features, 'label', double(logical(label)));
end

function candidate = localProbeWrongPCI(received, carrier, referenceCandidate, falsePCI)
sampleIndex = (0:(numel(received) - 1)).';
corrected = received .* exp(-1j * 2 * pi * referenceCandidate.cfo_hz * sampleIndex / localSampleRate(carrier));
slotLength = localSlotLength(carrier);
grid = nrOFDMDemodulate(carrier, corrected((referenceCandidate.timing_offset_samples + 1): ...
    (referenceCandidate.timing_offset_samples + slotLength)));
receivedPSS = grid(nrPSSIndices);
receivedSSS = grid(nrSSSIndices);
pss = nrPSS(mod(falsePCI, 3));
sss = nrSSS(falsePCI);
pssMetric = abs(sum(receivedPSS .* conj(pss))) / (norm(receivedPSS) * sqrt(numel(receivedPSS)) + eps);
sssMetric = abs(sum(receivedSSS .* conj(sss))) / (norm(receivedSSS) * sqrt(numel(receivedSSS)) + eps);
candidate = struct('pci', falsePCI, ...
    'timing_offset_samples', referenceCandidate.timing_offset_samples, ...
    'cfo_hz', referenceCandidate.cfo_hz, ...
    'correlation_metric', pssMetric + sssMetric, ...
    'pss_metric', pssMetric, 'sss_metric', sssMetric, ...
    'pss_peak_to_sidelobe', pssMetric / (1 - pssMetric + eps), ...
    'pss_sss_consistency', 1 - abs(pssMetric - sssMetric), ...
    'dmrs_gain_abs', 0, 'dmrs_gain_variance', 0, 'llr_mean_abs', 0, ...
    'pbch_crc_ok', false);
end

function [candidate, details] = localDetect(received, carrier, cfoRange, searchLimit, noiseVariance)
[offset, nid2, coarseCFO, pssMetric] = localSearchPSS(received, carrier, cfoRange, searchLimit);
slotLength = localSlotLength(carrier);
uncompensated = received((offset + 1):(offset + slotLength));
uncompensatedGrid = nrOFDMDemodulate(carrier, uncompensated);
[pci, sssMetric] = localSearchSSS(uncompensatedGrid, nid2);
cfo = localRefineCFO(uncompensatedGrid, carrier, pci, coarseCFO);
sampleIndex = (0:(numel(received) - 1)).';
corrected = received .* exp(-1j * 2 * pi * cfo * sampleIndex / localSampleRate(carrier));
grid = nrOFDMDemodulate(carrier, corrected((offset + 1):(offset + slotLength)));
dmrs = nrPBCHDMRS(pci, 0);
dmrsIndices = nrPBCHDMRSIndices(pci);
pbchIndices = nrPBCHIndices(pci);
[channelEstimate, estimatedNoiseVariance] = nrChannelEstimate(grid, dmrsIndices, dmrs);
equalizedPBCH = grid(pbchIndices) ./ channelEstimate(pbchIndices);
softBits = nrPBCHDecode(equalizedPBCH, pci, 0, max(estimatedNoiseVariance, noiseVariance));
[~, decodeError, transportBlock] = nrBCHDecode(softBits, 8, 4, pci);
channelGain = mean(channelEstimate(dmrsIndices));
pbchCRCOK = decodeError == 0;
candidate = struct('pci', pci, 'timing_offset_samples', offset, 'cfo_hz', cfo, ...
    'correlation_metric', pssMetric + sssMetric, 'pss_metric', pssMetric, ...
    'sss_metric', sssMetric, 'pss_peak_to_sidelobe', pssMetric / (1 - pssMetric + eps), ...
    'pss_sss_consistency', 1 - abs(pssMetric - sssMetric), ...
    'dmrs_gain_abs', abs(channelGain), 'dmrs_gain_variance', var(abs(channelEstimate(dmrsIndices))), ...
    'llr_mean_abs', mean(abs(softBits)), 'pbch_crc_ok', logical(pbchCRCOK));
details = struct('pbch_crc_ok', logical(pbchCRCOK), ...
    'transport_block', transportBlock, 'channel_gain', channelGain);
end

function [waveform, grid] = localBuildSSB(carrier, ncellid, trblk)
grid = nrResourceGrid(carrier);
grid(nrPSSIndices) = nrPSS(mod(ncellid, 3));
grid(nrSSSIndices) = nrSSS(ncellid);
codeword = nrBCH(double(trblk), 0, 0, 4, 0, ncellid);
grid(nrPBCHIndices(ncellid)) = nrPBCH(codeword, ncellid, 0);
grid(nrPBCHDMRSIndices(ncellid)) = nrPBCHDMRS(ncellid, 0);
waveform = nrOFDMModulate(carrier, grid);
end

function placed = localPlaceWaveform(waveform, offset, cfo, gain, totalLength, sampleRate)
placed = zeros(totalLength, 1);
indices = (offset + 1):(offset + numel(waveform));
phase = exp(1j * 2 * pi * cfo * ((indices - 1).') / sampleRate);
placed(indices) = gain .* waveform .* phase;
end

function [offset, nid2, cfo, peakMetric] = localSearchPSS(received, carrier, cfoRange, maxOffset)
emptyGrid = nrResourceGrid(carrier);
[~, info] = nrOFDMModulate(carrier, emptyGrid);
symbolLength = info.SymbolLengths(1);
searchEnd = min(maxOffset, numel(received) - symbolLength);
candidateCFOs = cfoRange(1):cfoRange(3):cfoRange(2);
bestMetric = -inf;
offset = 0; nid2 = 0; cfo = 0;
for candidateNID2 = 0:2
    pssGrid = nrResourceGrid(carrier);
    pssGrid(nrPSSIndices) = nrPSS(candidateNID2);
    pssWaveform = nrOFDMModulate(carrier, pssGrid);
    pssTemplate = pssWaveform(1:symbolLength);
    templateEnergy = norm(pssTemplate);
    for candidateCFO = candidateCFOs
        for candidateOffset = 0:searchEnd
            indices = (candidateOffset + 1):(candidateOffset + symbolLength);
            phase = exp(-1j * 2 * pi * candidateCFO * ((indices - 1).') / info.SampleRate);
            segment = received(indices) .* phase;
            metric = abs(sum(segment .* conj(pssTemplate))) / (norm(segment) * templateEnergy + eps);
            if metric > bestMetric
                bestMetric = metric;
                offset = candidateOffset;
                nid2 = candidateNID2;
                cfo = candidateCFO;
            end
        end
    end
end
peakMetric = bestMetric;
end

function [pci, metric] = localSearchSSS(grid, nid2)
receivedSSS = grid(nrSSSIndices);
bestMetric = -inf; pci = -1;
for nid1 = 0:335
    candidatePCI = 3 * nid1 + nid2;
    score = abs(sum(receivedSSS .* conj(nrSSS(candidatePCI)))) / ...
        (norm(receivedSSS) * sqrt(numel(receivedSSS)) + eps);
    if score > bestMetric
        bestMetric = score;
        pci = candidatePCI;
    end
end
metric = bestMetric;
end

function cfo = localRefineCFO(grid, carrier, pci, coarseCFO)
pssPhase = angle(sum(grid(nrPSSIndices) .* conj(nrPSS(mod(pci, 3)))));
sssPhase = angle(sum(grid(nrSSSIndices) .* conj(nrSSS(pci))));
ofdmInfo = nrOFDMInfo(carrier);
separationSeconds = sum(ofdmInfo.SymbolLengths(1:2)) / ofdmInfo.SampleRate;
phaseCFO = angle(exp(1j * (sssPhase - pssPhase))) / (2 * pi * separationSeconds);
cfo = phaseCFO + round((coarseCFO - phaseCFO) * separationSeconds) / separationSeconds;
end

function sampleRate = localSampleRate(carrier)
info = nrOFDMInfo(carrier);
sampleRate = info.SampleRate;
end

function slotLength = localSlotLength(carrier)
grid = nrResourceGrid(carrier);
waveform = nrOFDMModulate(carrier, grid);
slotLength = numel(waveform);
end

function localWriteJSON(payload, outputPath)
outDir = fileparts(outputPath);
if ~isempty(outDir) && ~isfolder(outDir)
    mkdir(outDir);
end
fid = fopen(outputPath, 'w');
assert(fid ~= -1, 'Cannot open result output path.');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(payload));
end

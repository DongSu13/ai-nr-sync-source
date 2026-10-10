function result = run_single_user_baseline(configPath, outputPath)
%RUN_SINGLE_USER_BASELINE Standard NR SSB acquisition with PBCH confirmation.
% Uses 38.211 PSS/SSS/PBCH resource mapping and 38.212 BCH Polar decoding.
if nargin < 1
    configPath = fullfile('configs', 'stage1_awgn.json');
end
if nargin < 2
    outputPath = fullfile('outputs', 'stage1_single_user.json');
end

cfg = jsondecode(fileread(configPath));
rng(cfg.seed, 'twister');
carrier = nrCarrierConfig;
carrier.NSizeGrid = cfg.nsize_grid;
carrier.SubcarrierSpacing = cfg.scs_khz;
carrier.NCellID = cfg.ncellid;

[tx, txGrid, transportBlock, ofdmInfo] = localBuildSSB(carrier, cfg.ncellid);
sampleRate = ofdmInfo.SampleRate;
timingOffset = cfg.timing_offset_samples;
pad = zeros(timingOffset, 1);
received = [pad; tx; zeros(128, 1)];
sampleIndex = (0:(numel(received) - 1)).';
received = received .* exp(1j * 2 * pi * cfg.cfo_hz * sampleIndex / sampleRate);
noiseVariance = mean(abs(tx).^2) / (10^(cfg.snr_db / 10));
received = received + sqrt(noiseVariance / 2) * (randn(size(received)) + 1j * randn(size(received)));

[estimatedOffset, estimatedNID2, coarseCFO, pssMetric] = ...
    localSearchPSS(received, carrier, cfg.cfo_search_hz, 1024);
uncompensatedWaveform = received((estimatedOffset + 1):(estimatedOffset + numel(tx)));
uncompensatedGrid = nrOFDMDemodulate(carrier, uncompensatedWaveform);
[estimatedPCI, sssMetric] = localSearchSSS(uncompensatedGrid, estimatedNID2);
estimatedCFO = localRefineCFO(uncompensatedGrid, carrier, estimatedPCI, coarseCFO);
corrected = received .* exp(-1j * 2 * pi * estimatedCFO * sampleIndex / sampleRate);
slotLength = numel(tx);
recoveredWaveform = corrected((estimatedOffset + 1):(estimatedOffset + slotLength));
rxGrid = nrOFDMDemodulate(carrier, recoveredWaveform);

pbchSymbols = rxGrid(nrPBCHIndices(estimatedPCI));
softBits = nrPBCHDecode(pbchSymbols, estimatedPCI, 0, noiseVariance);
[~, decodeError, decodedTransportBlock] = nrBCHDecode(softBits, 8, 4, estimatedPCI);
pbchCRCOK = decodeError == 0 && isequal(decodedTransportBlock, transportBlock);

candidate = struct( ...
    'pci', estimatedPCI, ...
    'timing_offset_samples', estimatedOffset, ...
    'cfo_hz', estimatedCFO, ...
    'correlation_metric', pssMetric + sssMetric, ...
    'pss_metric', pssMetric, ...
    'sss_metric', sssMetric, ...
    'pbch_crc_ok', logical(pbchCRCOK));
truth = struct( ...
    'pci', cfg.ncellid, ...
    'timing_offset_samples', timingOffset, ...
    'cfo_hz', cfg.cfo_hz, ...
    'pbch_transport_block', transportBlock.');
metrics = struct( ...
    'pci_match', logical(estimatedPCI == cfg.ncellid), ...
    'timing_error_samples', estimatedOffset - timingOffset, ...
    'cfo_error_hz', estimatedCFO - cfg.cfo_hz, ...
    'noise_variance', noiseVariance, ...
    'sample_rate_hz', sampleRate);
status = 'failure';
failureReason = 'candidate or PBCH recovery mismatch';
if metrics.pci_match && abs(metrics.timing_error_samples) <= 1 && ...
        abs(metrics.cfo_error_hz) <= 25 && candidate.pbch_crc_ok
    status = 'success';
    failureReason = '';
end
result = struct('stage', 'stage1', 'status', status, 'failure_reason', failureReason, ...
    'config', cfg, 'truth', truth, 'candidate', candidate, 'metrics', metrics);
localWriteJSON(result, outputPath);
end

function [waveform, grid, trblk, info] = localBuildSSB(carrier, ncellid)
grid = nrResourceGrid(carrier);
grid(nrPSSIndices) = nrPSS(mod(ncellid, 3));
grid(nrSSSIndices) = nrSSS(ncellid);
trblk = randi([0 1], 24, 1);
codeword = nrBCH(trblk, 0, 0, 4, 0, ncellid);
grid(nrPBCHIndices(ncellid)) = nrPBCH(codeword, ncellid, 0);
grid(nrPBCHDMRSIndices(ncellid)) = nrPBCHDMRS(ncellid, 0);
[waveform, info] = nrOFDMModulate(carrier, grid);
end

function [offset, nid2, cfo, peakMetric] = localSearchPSS(received, carrier, cfoRange, maxOffset)
emptyGrid = nrResourceGrid(carrier);
[~, info] = nrOFDMModulate(carrier, emptyGrid);
symbolLength = info.SymbolLengths(1);
searchEnd = min(maxOffset, numel(received) - symbolLength);
searchOffsets = 0:searchEnd;
candidateCFOs = cfoRange(1):cfoRange(3):cfoRange(2);
sampleRate = info.SampleRate;
bestMetric = -inf;
offset = 0;
nid2 = 0;
cfo = 0;
for candidateNID2 = 0:2
    pssGrid = nrResourceGrid(carrier);
    pssGrid(nrPSSIndices) = nrPSS(candidateNID2);
    pssWaveform = nrOFDMModulate(carrier, pssGrid);
    pssTemplate = pssWaveform(1:symbolLength);
    templateEnergy = norm(pssTemplate);
    for candidateCFO = candidateCFOs
        for candidateOffset = searchOffsets
            indices = (candidateOffset + 1):(candidateOffset + symbolLength);
            phase = exp(-1j * 2 * pi * candidateCFO * ((indices - 1).') / sampleRate);
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
bestMetric = -inf;
pci = -1;
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
% PSS and SSS have known sequences in symbols 1 and 3 of an SSB. Their
% correlation-phase difference provides a finer CFO estimate than a
% single-symbol PSS frequency grid can provide.
nid2 = mod(pci, 3);
pssPhase = angle(sum(grid(nrPSSIndices) .* conj(nrPSS(nid2))));
sssPhase = angle(sum(grid(nrSSSIndices) .* conj(nrSSS(pci))));
ofdmInfo = nrOFDMInfo(carrier);
separationSeconds = sum(ofdmInfo.SymbolLengths(1:2)) / ofdmInfo.SampleRate;
phaseCFO = angle(exp(1j * (sssPhase - pssPhase))) / (2 * pi * separationSeconds);

% Resolve phase wrapping to the alias nearest the coarse PSS-grid result.
aliasSpacing = 1 / separationSeconds;
cfo = phaseCFO + round((coarseCFO - phaseCFO) / aliasSpacing) * aliasSpacing;
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

function probe_5g_toolbox(outputPath)
% Write a machine-readable record of the MATLAB/5G Toolbox environment.
if nargin < 1
    outputPath = fullfile('outputs', 'environment_probe.json');
end

outDir = fileparts(outputPath);
if ~isempty(outDir) && ~isfolder(outDir)
    mkdir(outDir);
end

details = ver('5G_Toolbox');
payload = struct();
payload.matlab_version = version;
payload.nrWaveformGenerator_available = exist('nrWaveformGenerator', 'file') == 2;
payload.nrCellSearch_available = exist('nrCellSearch', 'file') == 2;
payload.nrPBCHDecode_available = exist('nrPBCHDecode', 'file') == 2;
payload.five_g_toolbox_installed = ~isempty(details);
payload.five_g_toolbox_version = '';
if ~isempty(details)
    payload.five_g_toolbox_version = details.Version;
end

fid = fopen(outputPath, 'w');
assert(fid ~= -1, 'Cannot open environment probe output.');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(payload));
end

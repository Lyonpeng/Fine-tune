const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');

function createRun(root, dataset) {
    const name = `${path.basename(dataset)}_${new Date().toISOString().replace(/[:.]/g, '-')}_${randomUUID()}`;
    const outputDir = path.join(root, name);
    fs.mkdirSync(outputDir, { recursive: true });
    return { name, outputDir };
}

function saveRun(job) {
    const { process, logs, outputDir, ...metadata } = job;
    const destination = path.join(outputDir, 'run.json');
    fs.writeFileSync(`${destination}.tmp`, JSON.stringify(metadata, null, 2));
    fs.renameSync(`${destination}.tmp`, destination);
}

function readHistory(root, liveJobs) {
    if (!fs.existsSync(root)) return liveJobs;
    const liveNames = new Set(liveJobs.map(job => job.name));
    const saved = [];
    for (const folder of fs.readdirSync(root, { withFileTypes: true })) {
        if (!folder.isDirectory() || liveNames.has(folder.name)) continue;
        const directory = path.join(root, folder.name);
        let metadata;
        try {
            metadata = JSON.parse(fs.readFileSync(path.join(directory, 'run.json'), 'utf8'));
        } catch (error) {
            if (error.code !== 'ENOENT') console.error(`Cannot read history for ${folder.name}:`, error.message);
        }
        if (metadata) {
            if (metadata.status === 'Failed') metadata.status = 'Error';
            if (['Running', 'Stopping'].includes(metadata.status)) {
                metadata.status = 'Unknown';
                metadata.endTime = null;
            }
            saved.push({ ...metadata, name: folder.name });
            continue;
        }
        const stats = fs.statSync(directory);
        let config = '';
        try { config = fs.readFileSync(path.join(directory, 'config.yaml'), 'utf8'); }
        catch (error) { if (error.code !== 'ENOENT') console.error(error.message); }
        const number = key => {
            const match = config.match(new RegExp(`^  ${key}: ([\\d.eE+-]+)`, 'm'));
            return match ? Number(match[1]) : null;
        };
        const dataset = config.match(/^  TRAIN:\s*\r?\n  - (.+)/m);
        saved.push({
            id: folder.name, name: folder.name,
            dataset: dataset ? dataset[1].replace(/^train_/, '') : folder.name,
            model: 'Mask R-CNN', learningRate: number('BASE_LR'),
            batchSize: number('IMS_PER_BATCH'), maxIterations: number('MAX_ITER'),
            status: fs.existsSync(path.join(directory, 'model_final.pth')) ? 'Completed' : 'Saved',
            startTime: stats.birthtimeMs, endTime: null, exitCode: null
        });
    }
    return [...liveJobs, ...saved].sort((a, b) => b.startTime - a.startTime);
}

module.exports = { createRun, saveRun, readHistory };

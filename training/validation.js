const fs = require('fs');
const path = require('path');

function createValidation(root) {
    fs.mkdirSync(root, { recursive: true });
    const id = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    const outputDir = path.join(root, id);
    fs.mkdirSync(outputDir);
    return { id, outputDir };
}

function saveValidation(job) {
    const { process, outputDir, metrics, ...metadata } = job;
    const destination = path.join(outputDir, 'validation.json');
    fs.writeFileSync(`${destination}.tmp`, JSON.stringify(metadata, null, 2));
    fs.renameSync(`${destination}.tmp`, destination);
}

function readMetrics(directory) {
    try {
        return JSON.parse(fs.readFileSync(path.join(directory, 'metrics.json'), 'utf8'));
    } catch (error) {
        if (error.code !== 'ENOENT') console.error(`Cannot read metrics in ${directory}:`, error.message);
        return null;
    }
}

// Validation jobs still running in this server, plus every one saved on disk
function readValidations(root, liveJobs) {
    const live = liveJobs.map(job => ({ ...job, metrics: job.status === 'Completed' ? readMetrics(job.outputDir) : null }));
    if (!fs.existsSync(root)) return live;
    const liveIds = new Set(liveJobs.map(job => job.id));
    const saved = [];
    for (const folder of fs.readdirSync(root, { withFileTypes: true })) {
        if (!folder.isDirectory() || liveIds.has(folder.name)) continue;
        const directory = path.join(root, folder.name);
        let metadata;
        try {
            metadata = JSON.parse(fs.readFileSync(path.join(directory, 'validation.json'), 'utf8'));
        } catch (error) {
            if (error.code !== 'ENOENT') console.error(`Cannot read validation ${folder.name}:`, error.message);
            continue;
        }
        // the server stopped while this validation was running
        if (metadata.status === 'Running') metadata.status = 'Unknown';
        saved.push({ ...metadata, id: folder.name, metrics: readMetrics(directory) });
    }
    return [...live, ...saved].sort((a, b) => b.startTime - a.startTime);
}

// Annotated images of a dataset folder that exist on disk, with their object counts.
// Like train_net.py, the folder must hold exactly one COCO annotation file.
function readDatasetImages(datasetPath) {
    const files = fs.readdirSync(datasetPath, { withFileTypes: true }).filter(file => file.isFile());
    const annotations = files.filter(file => file.name.toLowerCase().endsWith('.json'));
    if (annotations.length !== 1) {
        throw new Error(annotations.length
            ? 'The dataset has more than one JSON annotation file.'
            : 'The dataset has no JSON annotation file.');
    }
    const coco = JSON.parse(fs.readFileSync(path.join(datasetPath, annotations[0].name), 'utf8'));
    if (!Array.isArray(coco.images) || !Array.isArray(coco.annotations)) {
        throw new Error('The annotation file is not in COCO format.');
    }
    // annotations may differ in case from the files (e.g. 2.JPG vs 2.jpg);
    // validate.py resolves the real names
    const onDisk = new Set(files.map(file => file.name.toLowerCase()));
    const counts = new Map();
    for (const annotation of coco.annotations) {
        counts.set(annotation.image_id, (counts.get(annotation.image_id) || 0) + 1);
    }
    return {
        annotation: annotations[0].name,
        images: coco.images
            .filter(image => onDisk.has(String(image.file_name).toLowerCase()))
            .map(image => ({ fileName: image.file_name, objects: counts.get(image.id) || 0 }))
    };
}

module.exports = { createValidation, saveValidation, readValidations, readDatasetImages };

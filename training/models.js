const fs = require('fs');
const path = require('path');
const { readHistory } = require('./history');

function readModels(root, liveJobs = []) {
    return readHistory(root, liveJobs).flatMap(run => {
        const directory = path.join(root, run.name);
        if (!fs.existsSync(directory)) return [];
        return fs.readdirSync(directory, { withFileTypes: true })
            .filter(file => file.isFile() && file.name.endsWith('.pth'))
            .map(file => ({
                name: run.name,
                filename: file.name,
                model: run.model || 'Unknown',
                dataset: run.dataset || 'Unknown',
                completedAt: run.endTime || (run.status === 'Completed'
                    ? fs.statSync(path.join(directory, file.name)).mtimeMs : null),
                hasLoss: fs.existsSync(path.join(directory, 'training_loss.png')),
                hasConfig: fs.existsSync(path.join(directory, 'config.yaml'))
            }));
    }).sort((a, b) => (b.completedAt || 0) - (a.completedAt || 0)
        || a.name.localeCompare(b.name) || a.filename.localeCompare(b.filename));
}

module.exports = { readModels };

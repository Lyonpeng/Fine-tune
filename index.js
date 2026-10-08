const express = require('express');
const app = express();
const port = process.env.PORT || 5000;
// Python interpreter that has detectron2 (e.g. a conda env locally, "python" in Docker)
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const path = require('path');
const { spawn } = require("child_process");
const { createRun, saveRun, readHistory, isValidRunName, runNameExists } = require('./training/history');
const { readModels } = require('./training/models');
const trainingRoot = path.join(__dirname, 'outputs', 'training');
app.use(express.json());
let trainingJobs = [];
let nextRunId = 1;

app.get("/api/training", (req, res) => {
    const now = Date.now();

    const jobs = readHistory(trainingRoot, trainingJobs).map(job => ({
        id: job.id,
        name: job.name,
        dataset: job.dataset,
        model: job.model,
        learningRate: job.learningRate,
        batchSize: job.batchSize,
        maxIterations: job.maxIterations,
        weights: job.weights,
        status: job.status,
        error: job.error || null,
        startTime: job.startTime,
        endTime: job.endTime,
        exitCode: job.exitCode,
        runtime: ["Running", "Stopping"].includes(job.status)
            ? now - job.startTime
            : job.endTime ? job.endTime - job.startTime : null
    }));

    res.json(jobs);
});
app.get('/api/training/:name/config', (req, res) => {
    const name = req.params.name;
    if (!name || name === '.' || name === '..' || /[\\/]/.test(name)) {
        return res.status(400).type('text/plain').send('Invalid run name');
    }
    const configPath = path.join(trainingRoot, name, 'config.yaml');
    fs.readFile(configPath, 'utf8', (error, config) => {
        if (error) {
            return res.status(error.code === 'ENOENT' ? 404 : 500)
                .type('text/plain').send('Configuration is not available for this run.');
        }
        res.set('Cache-Control', 'no-store').type('text/plain').send(config);
    });
});
app.get('/api/training/:name/errors', (req, res) => {
    const name = req.params.name;
    if (!name || name === '.' || name === '..' || /[\\/]/.test(name)) {
        return res.status(400).type('text/plain').send('Invalid run name');
    }
    const job = readHistory(trainingRoot, trainingJobs).find(run => run.name === name);
    if (!job) return res.status(404).type('text/plain').send('Training run not found.');
    const details = job.error || (job.logs && job.logs.join(''))
        || 'No error details were saved for this run.';
    res.set('Cache-Control', 'no-store').type('text/plain').send(
        `Training run: ${job.name}\nStatus: ${job.status}\nExit code: ${job.exitCode ?? 'Unavailable'}\n\n${details}`
    );
});
app.get('/api/training/:name/loss', (req, res) => {
    const name = req.params.name;
    if (!name || name === '.' || name === '..' || /[\\/]/.test(name)) {
        return res.status(400).type('text/plain').send('Invalid run name');
    }
    res.sendFile(path.join(trainingRoot, name, 'training_loss.png'), error => {
        if (error && !res.headersSent) {
            res.status(error.statusCode || 500).type('text/plain')
                .send('Training loss is not available for this run.');
        }
    });
});
app.get('/api/training/:name/models/:filename', (req, res) => {
    const { name, filename } = req.params;
    if (!name || name === '.' || name === '..' || /[\\/]/.test(name)
        || /[\\/]/.test(filename) || !filename.endsWith('.pth')) {
        return res.status(400).type('text/plain').send('Invalid model path');
    }
    res.download(path.join(trainingRoot, name, filename), filename, error => {
        if (error && !res.headersSent) {
            res.status(error.statusCode || 500).type('text/plain')
                .send('Model is not available for this run.');
        }
    });
});
app.post("/api/training/:id/stop", (req, res) => {
    const runId = Number(req.params.id);

    const job = trainingJobs.find(job => job.id === runId);

    if (!job) {
        return res.status(404).json({
            message: "Training job not found"
        });
    }

    if (job.status !== "Running") {
        return res.status(400).json({
            message: "Training job is not running"
        });
    }

    if (!job.process) {
        return res.status(500).json({
            message: "Training process not found"
        });
    }

    job.status = "Stopping";

    const stopped = job.process.kill();

    if (!stopped) {
        job.status = "Running";

        return res.status(500).json({
            message: "Failed to stop training"
        });
    }

    res.json({
        message: `Training #${runId} is being stopped`
    });
});
// Set EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
// Home page route
app.get('/', (req, res) => {
  res.render('index');
});

// About page route
app.get('/about', (req, res) => {
  const data = {
    title: 'About Page',
    message: 'Learn more about us!'
  }
  res.render('about', { data });
});
app.get('/pages/data', (req, res) => {
  res.render('data');
});
app.get('/pages/train', (req, res) => {
  res.set('Cache-Control', 'no-store').render('train', {
    models: readModels(trainingRoot, trainingJobs)
  });
});
app.get('/pages/annotate', (req, res) => {
  res.render('annotate');
});
// Listen on the port
app.listen(port, () => console.log(`App listening on port ${port}`));

// file upload configuration
const multer = require("multer");
const fs = require("fs");
const ApiError = require("./public/utils/APIError");
const { StatusCodes } = require("http-status-codes");
const { isValidName, folderNameExists } = require("./utils/names");
const INVALID_NAME_MESSAGE = label =>
  `${label} must start with a letter or number and use only letters, numbers, spaces, '-', '_' or '.' (max 64 characters).`;

const uploadFiles = (
  main_folder_name,
  file_size_limit = 2000000000000,
  allowed_file_types = ["image/png", "image/jpg", "image/jpeg"]
) => {
  const MAIN_UPLOAD_FOLDER = path.join(__dirname, "public", main_folder_name);

  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      // every file of the request goes into the folder claimed for the first one
      if (req.datasetFolder) {
        return cb(null, req.datasetFolder);
      }

      const subFolderName = req.body.folderName;
      if (!isValidName(subFolderName)) {
        return cb(new ApiError(StatusCodes.BAD_REQUEST, INVALID_NAME_MESSAGE("Dataset name")));
      }

      const duplicate = new ApiError(
        StatusCodes.CONFLICT,
        `Dataset name "${subFolderName}" is already used. Please choose another name.`
      );
      if (folderNameExists(MAIN_UPLOAD_FOLDER, subFolderName)) {
        return cb(duplicate);
      }

      const folderPath = path.join(
        MAIN_UPLOAD_FOLDER,
        subFolderName
      );

      try {
        fs.mkdirSync(MAIN_UPLOAD_FOLDER, { recursive: true });
        // non-recursive: throws EEXIST if another upload claimed the name first
        fs.mkdirSync(folderPath);
      } catch (error) {
        return cb(error.code === 'EEXIST' ? duplicate : error);
      }

      req.datasetFolder = folderPath;
      cb(null, folderPath);
    },

    filename: (req, file, cb) => {
      cb(null, path.basename(file.originalname));
    }
  });

  return multer({
    storage,
    limits: {
      fileSize: file_size_limit
    },
    fileFilter: (req, file, cb) => {
      if (allowed_file_types.includes(file.mimetype)) {
        cb(null, true);
      } else {
        cb(
          new ApiError(
            StatusCodes.BAD_REQUEST,
            "The file format is not allowed!"
          )
        );
      }
    }
  });
};

const datasetUpload = uploadFiles("uploads", 2000000000000, ["image/png", "image/jpg", "image/jpeg", "application/json"]).fields([
    { name: "images", maxCount: 1000 },
    { name: "associatedData", maxCount: 1 }]);

app.post("/upload",
  // report upload errors (bad/duplicate name, file type) as JSON for the Data page
  (req, res, next) => datasetUpload(req, res, error => {
    if (error) {
      return res.status(error.statusCode || StatusCodes.BAD_REQUEST).json({ message: error.message });
    }
    next();
  }),
    (req, res) => {
    if (!req.files?.images?.length) {
      return res.status(StatusCodes.BAD_REQUEST).json({ message: "Please select at least one image." });
    }
    console.log(req.files);

    res.send({
      message: "Files uploaded successfully",
      files: req.files
    });
  }
);
const fsp = require("fs").promises;
app.delete("/api/datasets/:folderName", async (req, res) => {
    try {
        const folderName = req.params.folderName;
        const uploadPath = path.join(__dirname, "public", "uploads");
        const folderPath = path.join(uploadPath, folderName);

        if (path.dirname(folderPath) !== uploadPath) {
            return res.status(400).json({
                message: "Invalid folder name"
            });
        }

        try {
            await fsp.access(folderPath);
        } catch {
            return res.status(404).json({
                message: "Dataset not found"
            });
        }

        await fsp.rm(folderPath, {
            recursive: true,
            force: true
        });

        res.json({
            message: "Dataset deleted successfully"
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            message: "Failed to delete dataset"
        });
    }
});
app.get("/api/datasets", async (req, res) => {
  try {
    const uploadPath = path.join(__dirname, "public", "uploads");
    const folders = await fsp.readdir(uploadPath, { withFileTypes: true });

    const datasets = [];

    for (const folder of folders) {
      if (!folder.isDirectory()) continue;

      const folderPath = path.join(uploadPath, folder.name);
      const files = await fsp.readdir(folderPath, { withFileTypes: true });
      const stats = await fsp.stat(folderPath);

      const images = files.filter(file =>
        file.isFile() && /\.(jpg|jpeg|png)$/i.test(file.name)
      );

      const annotation = files.find(file =>
        file.isFile() && file.name.toLowerCase().endsWith(".json")
      );

      let totalSize = 0;

      for (const file of files) {
        if (!file.isFile()) continue;
        const fileStats = await fsp.stat(path.join(folderPath, file.name));
        totalSize += fileStats.size;
      }

      datasets.push({
        name: folder.name,
        imageCount: images.length,
        annotation: annotation ? annotation.name : null,
        created: stats.birthtime,
        size: totalSize
      });
    }

    res.json(datasets);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Failed to retrieve datasets" });
  }
});

app.post("/train", (req, res) => {
  const {
      runName,
      dataset,
      weights,
      learningRate,
      batchSize,
      maxIterations,
      advancedConfig
  } = req.body;

    if (
        typeof dataset !== 'string' || !dataset.trim() ||
        dataset === '.' || dataset === '..' || /[\\/:]/.test(dataset) ||
        !weights ||
        !Number.isFinite(learningRate) ||
        !Number.isInteger(batchSize) ||
        !Number.isInteger(maxIterations)
    ) {
        return res.status(400).json({
            message: "Invalid training configuration"
        });
    }

    if (!isValidRunName(runName)) {
        return res.status(400).json({
            message: INVALID_NAME_MESSAGE("Run name")
        });
    }

    if (runNameExists(trainingRoot, trainingJobs, runName)) {
        return res.status(409).json({
            message: `Run name "${runName}" is already used. Please choose another name.`
        });
    }
const datasetPath = path.join(
    __dirname,
    "public",
    "uploads",
    dataset
);

if (!fs.existsSync(datasetPath)) {
    return res.status(404).json({
        message: "Selected dataset does not exist"
    });
}
    let name, outputDir;
    try {
        ({ name, outputDir } = createRun(trainingRoot, runName));
    } catch (error) {
        if (error.code === 'EEXIST') {
            return res.status(409).json({
                message: `Run name "${runName}" is already used. Please choose another name.`
            });
        }
        throw error;
    }
    const runId = nextRunId++;
    const configFile = path.join(__dirname, 'training', 'mask_rcnn_R_50_FPN_1x_test.yaml');
    // Preserve the source configuration even if Python cannot initialize.
    fs.copyFileSync(configFile, path.join(outputDir, 'config.yaml'));

    const job = {
        id: runId,
        name,
        outputDir,
        model: "Mask R-CNN",
        dataset,
        learningRate,
        batchSize,
        maxIterations,
        weights,
        status: "Running",
        startTime: Date.now(),
        endTime: null,
        exitCode: null,
        logs: []
    };

    trainingJobs.unshift(job);
    saveRun(job);

    const args = [
        path.join(__dirname, "training", "train_net.py"),
        "--config-file",
        configFile,
        "--dataset",
        datasetPath,
        "--output-dir",
        outputDir,
        "--weights",
        weights,
        "--learning-rate",
        learningRate.toString(),
        "--batch-size",
        batchSize.toString(),
        "--max-iterations",
        maxIterations.toString()
    ];

    if (advancedConfig && advancedConfig.trim()) {
        args.push("--advanced-config", advancedConfig);
    }

    const pythonProcess = spawn(PYTHON_BIN, args, {
        cwd: __dirname
    });
    job.process = pythonProcess;
    let stderrTail = "";

    pythonProcess.stdout.on("data", data => {
        const output = data.toString();

        console.log(`[TRAIN #${runId}] ${output}`);

        job.logs.push(output);

        if (job.logs.length > 100) {
            job.logs.shift();
        }
    });

    pythonProcess.stderr.on("data", data => {
        const output = data.toString();
        stderrTail = (stderrTail + output).slice(-8000);

        console.error(`[TRAIN #${runId}] ${output}`);

        job.logs.push(output);

        if (job.logs.length > 100) {
            job.logs.shift();
        }
    });

    pythonProcess.on("error", error => {
        console.error(`Training #${runId} failed:`, error);

        job.status = "Error";
        job.error = error.message;
        job.endTime = Date.now();
        job.logs.push(error.message);
        saveRun(job);
    });

    pythonProcess.on("close", code => {
        console.log(`Training #${runId} finished with code ${code}`);

        job.exitCode = code;
        job.endTime = Date.now();
        job.process = null;

        if (job.status === "Stopping") {
            job.status = "Stopped";
        } else if (job.status !== "Error") {
            job.status = code === 0 ? "Completed" : "Error";
            if (code !== 0) {
                job.error = stderrTail.trim() || `Training exited with code ${code}`;
            }
        }
        saveRun(job);
    });

    res.status(202).json({
        message: "Training started",
        runId
    });
});

// ---------- Validation ----------
const { createValidation, saveValidation, readValidations, readDatasetImages } = require('./training/validation');
const validationRoot = path.join(__dirname, 'outputs', 'validation');
let validationJobs = [];

// path of an uploaded dataset folder, or null for an invalid / missing name
function datasetFolder(name) {
    if (typeof name !== 'string' || !name.trim() || name === '.' || name === '..' || /[\\/:]/.test(name)) {
        return null;
    }
    const folder = path.join(__dirname, 'public', 'uploads', name);
    return fs.existsSync(folder) && fs.statSync(folder).isDirectory() ? folder : null;
}

app.get('/pages/validation', (req, res) => {
    res.set('Cache-Control', 'no-store').render('validation', {
        models: readModels(trainingRoot, trainingJobs)
    });
});

app.get('/api/datasets/:name/images', (req, res) => {
    const folder = datasetFolder(req.params.name);
    if (!folder) {
        return res.status(404).json({ message: 'Dataset not found' });
    }
    try {
        res.json(readDatasetImages(folder));
    } catch (error) {
        res.status(422).json({ message: error.message });
    }
});

app.get('/api/validation', (req, res) => {
    const now = Date.now();
    res.json(readValidations(validationRoot, validationJobs).map(({ process, outputDir, ...job }) => ({
        ...job,
        runtime: job.status === 'Running' ? now - job.startTime
            : job.endTime ? job.endTime - job.startTime : null
    })));
});

app.post('/api/validation', (req, res) => {
    const { run, filename, dataset, images } = req.body;

    // only models listed under Available Models on the Train page can be validated
    const model = readModels(trainingRoot, trainingJobs)
        .find(item => item.name === run && item.filename === filename);
    if (!model) {
        return res.status(404).json({ message: 'Selected model does not exist' });
    }
    if (!model.hasConfig) {
        return res.status(422).json({ message: 'The selected model has no training configuration (config.yaml).' });
    }

    const datasetPath = datasetFolder(dataset);
    if (!datasetPath) {
        return res.status(404).json({ message: 'Selected dataset does not exist' });
    }

    let available;
    try {
        available = new Set(readDatasetImages(datasetPath).images.map(image => image.fileName));
    } catch (error) {
        return res.status(422).json({ message: error.message });
    }
    const selected = Array.isArray(images) ? [...new Set(images)] : [];
    if (!selected.length || !selected.every(image => available.has(image))) {
        return res.status(400).json({ message: 'Please select validation images from the dataset.' });
    }

    const { id, outputDir } = createValidation(validationRoot);
    const imagesFile = path.join(outputDir, 'images.json');
    fs.writeFileSync(imagesFile, JSON.stringify(selected));

    const job = {
        id,
        outputDir,
        run: model.name,
        filename: model.filename,
        model: model.model,
        trainedOn: model.dataset,
        dataset,
        imageCount: selected.length,
        status: 'Running',
        progress: null,
        error: null,
        startTime: Date.now(),
        endTime: null,
        exitCode: null
    };
    validationJobs.unshift(job);
    saveValidation(job);

    const runDir = path.join(trainingRoot, model.name);
    const pythonProcess = spawn(PYTHON_BIN, [
        path.join(__dirname, 'training', 'validate.py'),
        '--config-file', path.join(runDir, 'config.yaml'),
        '--weights', path.join(runDir, model.filename),
        '--dataset', datasetPath,
        '--images-file', imagesFile,
        '--output-dir', outputDir
    ], { cwd: __dirname });
    job.process = pythonProcess;
    let stderrTail = '';

    pythonProcess.stdout.on('data', data => {
        const output = data.toString();
        console.log(`[VALIDATE ${id}] ${output}`);
        const progress = [...output.matchAll(/^PROGRESS (\d+)\/(\d+)/gm)].pop();
        if (progress) {
            job.progress = { done: Number(progress[1]), total: Number(progress[2]) };
        }
    });

    pythonProcess.stderr.on('data', data => {
        const output = data.toString();
        stderrTail = (stderrTail + output).slice(-8000);
        console.error(`[VALIDATE ${id}] ${output}`);
    });

    pythonProcess.on('error', error => {
        console.error(`Validation ${id} failed:`, error);
        job.status = 'Error';
        job.error = error.message;
        job.endTime = Date.now();
        saveValidation(job);
    });

    pythonProcess.on('close', code => {
        console.log(`Validation ${id} finished with code ${code}`);
        job.exitCode = code;
        job.endTime = Date.now();
        job.process = null;
        if (job.status !== 'Error') {
            job.status = code === 0 ? 'Completed' : 'Error';
            if (code !== 0) {
                job.error = stderrTail.trim() || `Validation exited with code ${code}`;
            }
        }
        saveValidation(job);
    });

    res.status(202).json({ message: 'Validation started', id });
});
async function loadTrainingHistory() {
    const table = document.getElementById("runtimeHistory");

    if (!table) return;

    try {
        const response = await fetch("/api/training");

        if (!response.ok) {
            throw new Error("Failed to retrieve training history");
        }

        const jobs = await response.json();

        if (jobs.length === 0) {
            table.innerHTML = `
                <tr>
                    <td colspan="7">No training runs.</td>
                </tr>
            `;
            return;
        }

        table.innerHTML = "";

        jobs.forEach(job => {
            const row = document.createElement("tr");

            row.innerHTML = `
                <td>#${String(job.id).padStart(3, "0")}</td>
                <td>${job.model}</td>
                <td>${job.learningRate}</td>
                <td>${job.batchSize}</td>
                <td>${job.maxIterations.toLocaleString()}</td>
                <td>
                    <span class="status ${job.status.toLowerCase()}">
                        ${job.status}
                    </span>
                </td>
                <td>${formatRuntime(job.runtime)}</td>
            `;

            table.appendChild(row);
        });
    } catch (error) {
        console.error("Training history error:", error);
    }
}

function formatRuntime(milliseconds) {
    if (!milliseconds) return "0s";

    const totalSeconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }

    if (minutes > 0) {
        return `${minutes}m ${seconds}s`;
    }

    return `${seconds}s`;
}

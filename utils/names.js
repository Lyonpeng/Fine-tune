const fs = require('fs');

// Run and dataset names become folder names. Letters/digits first, then letters,
// digits, spaces, "-", "_" or "."; no trailing space or dot (Windows strips those).
const NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$/;

function isValidName(name) {
    return typeof name === 'string' && NAME_PATTERN.test(name) && !/[ .]$/.test(name);
}

// Case-insensitive, since Windows folder names ignore case.
function folderNameExists(directory, name) {
    if (!fs.existsSync(directory)) return false;
    const wanted = name.toLowerCase();
    return fs.readdirSync(directory).some(folder => folder.toLowerCase() === wanted);
}

module.exports = { isValidName, folderNameExists };

const fileInput = document.querySelector('#image-input');
const dropZone = document.querySelector('#drop-zone');
const browseButton = document.querySelector('#browse-button');
const emptyUpload = document.querySelector('#empty-upload');
const previewWrap = document.querySelector('#image-preview-wrap');
const preview = document.querySelector('#image-preview');
const fileName = document.querySelector('#image-file-name');
const removeImage = document.querySelector('#remove-image');
const fileError = document.querySelector('#file-error');
const form = document.querySelector('#vision-form');
const question = document.querySelector('#question');
const questionCount = document.querySelector('#question-count');
const questionError = document.querySelector('#question-error');
const analyzeButton = document.querySelector('#analyze-button');
const buttonLabel = document.querySelector('#button-label');
const resultEmpty = document.querySelector('#result-empty');
const resultLoading = document.querySelector('#result-loading');
const resultSuccess = document.querySelector('#result-success');
const resultError = document.querySelector('#result-error');
const answerText = document.querySelector('#answer-text');
const confidenceValue = document.querySelector('#confidence-value');
const confidenceBar = document.querySelector('#confidence-bar');
const newQuestion = document.querySelector('#new-question');
const statusMessage = document.querySelector('#status-message');

let selectedFile = null;
let previewUrl = null;
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

function announce(message) { statusMessage.textContent = message; }
function show(element) { element.hidden = false; }
function hide(element) { element.hidden = true; }
function setResult(state) {
  [resultEmpty, resultLoading, resultSuccess, resultError].forEach(hide);
  show(state);
}
function clearError(element) { hide(element); element.textContent = ''; }

function handleFile(file) {
  clearError(fileError);
  if (!file) return;
  if (!ACCEPTED_TYPES.includes(file.type)) {
    fileError.textContent = 'Please choose a JPG, PNG, or WEBP image.';
    show(fileError);
    announce('The selected file type is not supported.');
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    fileError.textContent = 'That image is larger than 10MB. Please choose a smaller file.';
    show(fileError);
    announce('The selected image is too large.');
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.alt = `Preview of ${file.name}`;
  fileName.textContent = file.name;
  hide(emptyUpload);
  show(previewWrap);
  announce(`${file.name} selected.`);
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.removeAttribute('src');
  hide(previewWrap);
  show(emptyUpload);
  clearError(fileError);
  announce('Image removed.');
}

function updateQuestionCount() {
  questionCount.textContent = `${question.value.length} / 240`;
}

function validate() {
  let valid = true;
  clearError(fileError);
  clearError(questionError);
  if (!selectedFile) {
    fileError.textContent = 'Please choose an image first.';
    show(fileError);
    valid = false;
  }
  if (!question.value.trim()) {
    questionError.textContent = 'Please add a question before continuing.';
    show(questionError);
    valid = false;
  }
  return valid;
}

const DEFAULT_ERROR_TEXT = 'Check your image and question, then try again.';

async function submitQuestion() {
  if (!validate()) {
    announce('Please complete the image and question fields.');
    return;
  }
  hide(resultEmpty);
  hide(resultSuccess);
  hide(resultError);
  show(resultLoading);
  analyzeButton.disabled = true;
  buttonLabel.textContent = 'Analyzing...';
  announce('Analyzing your image.');

  try {
    // Send the image and the question to the model server (POST /api/predict).
    const body = new FormData();
    body.append('image', selectedFile);
    body.append('question', question.value.trim());
    const response = await fetch('/api/predict', { method: 'POST', body });
    if (!response.ok) {
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail || DEFAULT_ERROR_TEXT);
    }
    const result = await response.json();

    // The server returns the model's softmax probability (0 to 1).
    const percent = Math.round(result.confidence * 1000) / 10;
    answerText.textContent = result.answer;
    confidenceValue.textContent = `${percent}%`;
    confidenceBar.style.width = `${Math.min(percent, 100)}%`;
    hide(resultLoading);
    show(resultSuccess);
    announce(`Analysis complete. Answer: ${result.answer}. Confidence ${percent} percent.`);
  } catch (error) {
    hide(resultLoading);
    resultError.querySelector('small').textContent = error.message || DEFAULT_ERROR_TEXT;
    show(resultError);
    announce('We could not process that request.');
  } finally {
    analyzeButton.disabled = false;
    buttonLabel.textContent = 'Analyze image';
  }
}

browseButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
removeImage.addEventListener('click', (event) => { event.stopPropagation(); clearFile(); });
dropZone.addEventListener('click', (event) => {
  if (event.target.closest('button')) return;
  if (!selectedFile) fileInput.click();
});
dropZone.addEventListener('keydown', (event) => {
  if ((event.key === 'Enter' || event.key === ' ') && !selectedFile) { event.preventDefault(); fileInput.click(); }
});
['dragenter', 'dragover'].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.add('is-dragging'); }));
['dragleave', 'drop'].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove('is-dragging'); }));
dropZone.addEventListener('drop', (event) => handleFile(event.dataTransfer.files[0]));
question.addEventListener('input', updateQuestionCount);
form.addEventListener('submit', (event) => { event.preventDefault(); submitQuestion(); });
newQuestion.addEventListener('click', () => { question.focus(); question.select(); });
document.querySelectorAll('.example-chip').forEach((chip) => chip.addEventListener('click', () => { question.value = chip.dataset.question; updateQuestionCount(); question.focus(); }));

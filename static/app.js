/**
 * 3D Extractor Frontend Application
 * Handles image upload, click-based segmentation, and 3D generation
 */

class App {
    constructor() {
        // State
        this.sessionId = null;
        this.imageLoaded = false;
        this.masks = [];
        this.scores = [];
        this.selectedMaskIndex = 0;
        this.clickPoints = [];
        this.currentTool = 'fg'; // 'fg' or 'bg'
        this.isProcessing = false;
        this.plyReady = false;
        
        // DOM Elements
        this.uploadZone = document.getElementById('upload-zone');
        this.canvasWrapper = document.getElementById('canvas-wrapper');
        this.mainCanvas = document.getElementById('main-canvas');
        this.maskOverlay = document.getElementById('mask-overlay');
        this.clickIndicator = document.getElementById('click-indicator');
        this.fileInput = document.getElementById('file-input');
        this.maskOptions = document.getElementById('mask-options');
        this.clickPointsDiv = document.getElementById('click-points');
        this.pointsList = document.getElementById('points-list');
        this.btnGenerate = document.getElementById('btn-generate');
        this.btnDownload = document.getElementById('btn-download');
        this.btnReset = document.getElementById('btn-reset');
        this.statusText = document.getElementById('status-text');
        
        // Canvas contexts
        this.ctx = this.mainCanvas.getContext('2d');
        this.maskCtx = this.maskOverlay.getContext('2d');
        
        // Image data
        this.originalImage = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.displayScale = 1;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkServerStatus();
    }
    
    setupEventListeners() {
        // File upload
        this.uploadZone.addEventListener('click', () => this.fileInput.click());
        this.uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadZone.addEventListener('dragleave', () => this.uploadZone.classList.remove('dragover'));
        this.uploadZone.addEventListener('drop', (e) => this.handleDrop(e));
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Canvas clicks
        this.canvasWrapper.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvasWrapper.addEventListener('contextmenu', (e) => this.handleCanvasRightClick(e));
        
        // Tool buttons
        document.getElementById('tool-fg').addEventListener('click', () => this.setTool('fg'));
        document.getElementById('tool-bg').addEventListener('click', () => this.setTool('bg'));
        document.getElementById('tool-clear').addEventListener('click', () => this.clearPoints());
        
        // Action buttons
        this.btnGenerate.addEventListener('click', () => this.generate3D());
        this.btnDownload.addEventListener('click', () => this.downloadPLY());
        this.btnReset.addEventListener('click', () => this.reset());
    }
    
    async checkServerStatus() {
        try {
            const response = await fetch('/api/info');
            const data = await response.json();
            
            const sam3Status = data.models.sam3.available ? 'SAM3 Ready' : 'SAM3 (Mock)';
            const sam3dStatus = data.models.sam3d.available ? 'SAM3D Ready' : 'SAM3D (Mock)';
            
            this.statusText.textContent = `${sam3Status} | ${sam3dStatus}`;
            
            if (!data.models.sam3.available || !data.models.sam3d.available) {
                this.showToast('Running in mock mode. Install SAM3 and SAM 3D Objects for full functionality.', 'warning');
            }
        } catch (error) {
            this.statusText.textContent = 'Server Error';
            this.showToast('Failed to connect to server', 'error');
        }
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.uploadZone.classList.add('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadZone.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            this.uploadImage(file);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.uploadImage(file);
        }
    }
    
    async uploadImage(file) {
        this.setProcessing(true);
        this.updateStep(1, 'active');
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/segment/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Upload failed');
            }
            
            const data = await response.json();
            this.sessionId = data.session_id;
            this.imageWidth = data.image_size.width;
            this.imageHeight = data.image_size.height;
            
            // Load image to canvas
            await this.loadImageToCanvas(file);
            
            this.imageLoaded = true;
            this.updateStep(1, 'completed');
            this.updateStep(2, 'active');
            
            this.showToast('Image uploaded successfully', 'success');
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast('Failed to upload image', 'error');
        } finally {
            this.setProcessing(false);
        }
    }
    
    async loadImageToCanvas(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    this.originalImage = img;
                    
                    // Calculate display size (max 800px)
                    const maxSize = 800;
                    let width = img.width;
                    let height = img.height;
                    
                    if (width > maxSize || height > maxSize) {
                        if (width > height) {
                            this.displayScale = maxSize / width;
                        } else {
                            this.displayScale = maxSize / height;
                        }
                        width = Math.round(width * this.displayScale);
                        height = Math.round(height * this.displayScale);
                    } else {
                        this.displayScale = 1;
                    }
                    
                    // Set canvas sizes
                    this.mainCanvas.width = width;
                    this.mainCanvas.height = height;
                    this.maskOverlay.width = width;
                    this.maskOverlay.height = height;
                    
                    // Draw image
                    this.ctx.drawImage(img, 0, 0, width, height);
                    
                    // Show canvas, hide upload zone
                    this.uploadZone.style.display = 'none';
                    this.canvasWrapper.style.display = 'block';
                    
                    resolve();
                };
                img.onerror = reject;
                img.src = e.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
    
    handleCanvasClick(e) {
        if (!this.imageLoaded || this.isProcessing) return;
        
        const rect = this.mainCanvas.getBoundingClientRect();
        const x = Math.round((e.clientX - rect.left) / this.displayScale);
        const y = Math.round((e.clientY - rect.top) / this.displayScale);
        
        // Add click point
        const label = this.currentTool === 'fg' ? 1 : 0;
        this.addClickPoint(x, y, label);
        
        // Show click indicator
        this.showClickIndicator(e.clientX - rect.left, e.clientY - rect.top);
        
        // Predict mask
        this.predictMask();
    }
    
    handleCanvasRightClick(e) {
        e.preventDefault();
        if (!this.imageLoaded || this.isProcessing) return;
        
        const rect = this.mainCanvas.getBoundingClientRect();
        const x = Math.round((e.clientX - rect.left) / this.displayScale);
        const y = Math.round((e.clientY - rect.top) / this.displayScale);
        
        // Add background point
        this.addClickPoint(x, y, 0);
        
        // Show click indicator
        this.showClickIndicator(e.clientX - rect.left, e.clientY - rect.top);
        
        // Predict mask
        this.predictMask();
    }
    
    addClickPoint(x, y, label) {
        this.clickPoints.push({ x, y, label });
        this.updatePointsDisplay();
    }
    
    updatePointsDisplay() {
        if (this.clickPoints.length === 0) {
            this.clickPointsDiv.style.display = 'none';
            return;
        }
        
        this.clickPointsDiv.style.display = 'block';
        this.pointsList.innerHTML = this.clickPoints.map((p, i) => `
            <span class="point-tag">
                <span class="dot ${p.label === 1 ? 'fg' : 'bg'}"></span>
                (${p.x}, ${p.y})
            </span>
        `).join('');
        
        // Draw points on canvas
        this.drawClickPoints();
    }
    
    drawClickPoints() {
        // Redraw image
        this.ctx.drawImage(this.originalImage, 0, 0, this.mainCanvas.width, this.mainCanvas.height);
        
        // Draw points
        this.clickPoints.forEach(p => {
            const x = p.x * this.displayScale;
            const y = p.y * this.displayScale;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, 2 * Math.PI);
            this.ctx.fillStyle = p.label === 1 ? '#00ff88' : '#ff4466';
            this.ctx.fill();
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
    }
    
    showClickIndicator(x, y) {
        this.clickIndicator.style.left = x + 'px';
        this.clickIndicator.style.top = y + 'px';
        this.clickIndicator.style.display = 'block';
        
        // Restart animation
        this.clickIndicator.style.animation = 'none';
        this.clickIndicator.offsetHeight; // Trigger reflow
        this.clickIndicator.style.animation = 'clickPulse 0.5s ease-out';
        
        setTimeout(() => {
            this.clickIndicator.style.display = 'none';
        }, 500);
    }
    
    setTool(tool) {
        this.currentTool = tool;
        document.getElementById('tool-fg').classList.toggle('active', tool === 'fg');
        document.getElementById('tool-bg').classList.toggle('active', tool === 'bg');
    }
    
    clearPoints() {
        this.clickPoints = [];
        this.masks = [];
        this.updatePointsDisplay();
        this.clearMaskOverlay();
        this.updateMaskOptions();
        this.btnGenerate.disabled = true;
        
        // Redraw image
        if (this.originalImage) {
            this.ctx.drawImage(this.originalImage, 0, 0, this.mainCanvas.width, this.mainCanvas.height);
        }
    }
    
    clearMaskOverlay() {
        this.maskCtx.clearRect(0, 0, this.maskOverlay.width, this.maskOverlay.height);
    }
    
    async predictMask() {
        if (this.clickPoints.length === 0 || !this.sessionId) return;
        
        this.setProcessing(true);
        
        try {
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('points_x', this.clickPoints.map(p => p.x).join(','));
            formData.append('points_y', this.clickPoints.map(p => p.y).join(','));
            formData.append('labels', this.clickPoints.map(p => p.label).join(','));
            
            const response = await fetch('/api/segment/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const data = await response.json();
            this.masks = data.masks;
            this.scores = data.scores;
            this.selectedMaskIndex = 0;
            
            // Update UI
            this.updateMaskOptions();
            this.displayMask(0);
            
            this.btnGenerate.disabled = false;
            this.updateStep(2, 'completed');
            this.updateStep(3, 'active');
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showToast('Failed to predict mask', 'error');
        } finally {
            this.setProcessing(false);
        }
    }
    
    updateMaskOptions() {
        if (this.masks.length === 0) {
            this.maskOptions.innerHTML = `
                <p style="color: var(--text-muted); font-size: 0.9rem;">
                    Click on the image to generate masks
                </p>
            `;
            return;
        }
        
        const colors = ['#0078ff', '#00ff78', '#ff7800'];
        
        this.maskOptions.innerHTML = this.masks.map((mask, i) => `
            <div class="mask-option ${i === this.selectedMaskIndex ? 'selected' : ''}" data-index="${i}">
                <div class="mask-preview">
                    <img src="data:image/png;base64,${mask}" alt="Mask ${i + 1}">
                </div>
                <div class="mask-info">
                    <div class="label">Mask ${i + 1}</div>
                    <div class="score" style="color: ${colors[i % colors.length]}">
                        Score: ${(this.scores[i] * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `).join('');
        
        // Add click handlers
        this.maskOptions.querySelectorAll('.mask-option').forEach(el => {
            el.addEventListener('click', () => {
                const index = parseInt(el.dataset.index);
                this.selectMask(index);
            });
        });
    }
    
    selectMask(index) {
        this.selectedMaskIndex = index;
        
        // Update UI
        this.maskOptions.querySelectorAll('.mask-option').forEach((el, i) => {
            el.classList.toggle('selected', i === index);
        });
        
        // Display selected mask
        this.displayMask(index);
    }
    
    displayMask(index) {
        if (!this.masks[index]) return;
        
        const img = new Image();
        img.onload = () => {
            this.clearMaskOverlay();
            this.maskCtx.drawImage(img, 0, 0, this.maskOverlay.width, this.maskOverlay.height);
        };
        img.src = 'data:image/png;base64,' + this.masks[index];
    }
    
    async generate3D() {
        if (!this.sessionId || this.masks.length === 0) return;
        
        this.setProcessing(true);
        this.btnGenerate.innerHTML = '<span class="spinner"></span> Generating...';
        
        try {
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('mask_index', this.selectedMaskIndex);
            formData.append('seed', 42);
            
            const response = await fetch('/api/reconstruct/generate', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('3D generation failed');
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.plyReady = true;
                this.btnDownload.style.display = 'flex';
                this.updateStep(3, 'completed');
                
                const msg = data.mock 
                    ? '3D generated (mock mode). Install SAM 3D Objects for real results.'
                    : '3D Gaussian Splat generated successfully!';
                this.showToast(msg, 'success');
            }
            
        } catch (error) {
            console.error('Generation error:', error);
            this.showToast('Failed to generate 3D', 'error');
        } finally {
            this.setProcessing(false);
            this.btnGenerate.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                    <path d="M2 17l10 5 10-5"></path>
                    <path d="M2 12l10 5 10-5"></path>
                </svg>
                Generate Gaussian Splat
            `;
        }
    }
    
    downloadPLY() {
        if (!this.sessionId || !this.plyReady) return;
        
        window.location.href = `/api/reconstruct/download/${this.sessionId}`;
        this.showToast('Downloading PLY file...', 'success');
    }
    
    reset() {
        // Reset state
        this.sessionId = null;
        this.imageLoaded = false;
        this.masks = [];
        this.scores = [];
        this.selectedMaskIndex = 0;
        this.clickPoints = [];
        this.plyReady = false;
        this.originalImage = null;
        
        // Reset UI
        this.uploadZone.style.display = 'flex';
        this.canvasWrapper.style.display = 'none';
        this.btnGenerate.disabled = true;
        this.btnDownload.style.display = 'none';
        this.clickPointsDiv.style.display = 'none';
        this.fileInput.value = '';
        
        // Clear canvases
        this.ctx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height);
        this.maskCtx.clearRect(0, 0, this.maskOverlay.width, this.maskOverlay.height);
        
        // Reset mask options
        this.maskOptions.innerHTML = `
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                Click on the image to generate masks
            </p>
        `;
        
        // Reset steps
        this.updateStep(1, 'active');
        this.updateStep(2, 'pending');
        this.updateStep(3, 'pending');
        
        this.showToast('Reset complete', 'success');
    }
    
    updateStep(stepNum, status) {
        const step = document.getElementById(`step-${stepNum}`);
        step.classList.remove('active', 'completed');
        if (status === 'active') {
            step.classList.add('active');
        } else if (status === 'completed') {
            step.classList.add('completed');
        }
    }
    
    setProcessing(processing) {
        this.isProcessing = processing;
        this.statusText.textContent = processing ? 'Processing...' : 'Ready';
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
        toast.innerHTML = `
            <span style="font-size: 1.2rem;">${icon}</span>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});


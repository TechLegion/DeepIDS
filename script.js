// ===== DeepIDS v2.0 - Gradient Boosting Integrated Backend =====

// Auto-detect API base: relative in prod (Railway), localhost in dev
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : '';

let probabilityChartInstance = null;

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('detect-form');
    const csvUpload = document.getElementById('csv-upload');

    // Form submission handler
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        detectIntrusion();
    });

    // CSV Upload Handler
    if (csvUpload) {
        csvUpload.addEventListener('change', function(e) {
            handleCSVUpload(this);
        });
    }
});

// ============================================
// FEATURE EXTRACTION
// ============================================
function getFeatures() {
    return {
        protocol_type: document.getElementById('protocol').value,
        service: document.getElementById('service').value,
        flag: document.getElementById('flag').value,
        duration: parseFloat(document.getElementById('duration').value) || 0,
        src_bytes: parseFloat(document.getElementById('src_bytes').value) || 0,
        dst_bytes: parseFloat(document.getElementById('dst_bytes').value) || 0,
        count: parseFloat(document.getElementById('count').value) || 0,
        serror_rate: parseFloat(document.getElementById('serror_rate').value) || 0,
        dst_host_count: parseFloat(document.getElementById('dst_host_count').value) || 0
    };
}

// ============================================
// DETECTION LOGIC
// ============================================
async function detectIntrusion() {
    const features = getFeatures();
    const resultContainer = document.getElementById('result-container');
    const chartContainer = document.getElementById('chart-container');
    
    // Show Loading State
    chartContainer.style.display = 'none';
    resultContainer.innerHTML = `
        <div class="placeholder-state">
            <div class="radar-scan"></div>
            <h3 style="color: var(--accent);">Processing Network Telemetry...</h3>
            <p>Running highly-optimized Gradient Boosting inference model</p>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(features)
        });

        if (!response.ok) {
            throw new Error('Backend responded with an error');
        }

        const result = await response.json();
        displayResult(result, features);
    } catch (error) {
        console.error(error);
        resultContainer.innerHTML = `
            <div class="result-header" style="border-left: 4px solid var(--danger);">
                <div class="result-title">
                    <span style="font-size: 2rem;">❌</span>
                    <div>
                        <h3 style="color: var(--danger);">Connection Error</h3>
                        <p>Could not reach the FastAPI Backend. Ensure uvicorn is running.</p>
                    </div>
                </div>
            </div>
        `;
    }
}

// ============================================
// DISPLAY PREMIUM RESULT UI
// ============================================
function displayResult(result, features) {
    const resultContainer = document.getElementById('result-container');
    const isAttack = result.prediction !== 'normal';
    const confidence = Math.round(result.confidence * 100);
    const timestamp = new Date().toLocaleString();

    const statusIcon = isAttack ? '⚠️' : '✅';
    const statusColor = isAttack ? 'var(--danger)' : 'var(--success)';
    const statusGlow = isAttack ? 'var(--danger-glow)' : 'var(--success-glow)';

    const traceItems = isAttack 
        ? ['Anomalous packet signature identified', `Model: ${result.description || 'Gradient Boosting threat category classification'}`]
        : ['Feature vector aligns with baseline normal traffic', 'No known attack signatures identified'];

    let html = `
        <div class="result-header" style="border-left: 4px solid ${statusColor}; box-shadow: 0 0 20px ${statusGlow};">
            <div class="result-title">
                <span style="font-size: 2.5rem;">${statusIcon}</span>
                <div>
                    <h3 style="color: ${statusColor};">${result.label || result.prediction.toUpperCase()}</h3>
                    <p style="margin-top: 5px;">AI Confidence: <span style="font-weight: 800; color: #fff;">${confidence}%</span></p>
                </div>
            </div>
            <div class="threat-meter-bg">
                <div class="threat-meter-fill" style="width: 0%; background: ${statusColor};" id="animated-meter"></div>
            </div>
        </div>
        
        <div class="analysis-trace">
            <h4><span style="color: ${statusColor}">■</span> Threat Intelligence Trace</h4>
            <div class="trace-list">
                ${traceItems.map(r => `
                    <div class="trace-item">
                        <span class="bullet" style="background:${statusColor}"></span>
                        ${r}
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="recommendation-box" style="border-color: ${statusColor}; box-shadow: inset 0 0 20px ${statusGlow};">
            <div class="rec-label">RECOMMENDED SYSTEM ACTION</div>
            <div class="rec-action" style="color: ${statusColor}">
                ${isAttack ? 'BLOCK CONNECTION & LOG IP' : 'ALLOW CONNECTION'}
            </div>
        </div>

        <div class="scan-timestamp">Scan completed at ${timestamp}</div>
    `;

    resultContainer.innerHTML = html;

    // Animate the threat meter
    setTimeout(() => {
        const meter = document.getElementById('animated-meter');
        if (meter) meter.style.width = `${confidence}%`;
    }, 50);

    // --- Chart.js Integration ---
    const chartContainer = document.getElementById('chart-container');
    if (result.probabilities) {
        chartContainer.style.display = 'block';
        const ctx = document.getElementById('probabilityChart').getContext('2d');
        
        if (probabilityChartInstance) {
            probabilityChartInstance.destroy();
        }

        const labels = Object.keys(result.probabilities).map(l => l.toUpperCase());
        const data = Object.values(result.probabilities).map(v => Math.round(v * 100));

        probabilityChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: labels.map(l => {
                        if (l === 'NORMAL') return 'rgba(0, 184, 148, 0.8)';
                        if (l === 'DOS') return 'rgba(255, 118, 117, 0.8)';
                        if (l === 'PROBE') return 'rgba(253, 203, 110, 0.8)';
                        if (l === 'R2L') return 'rgba(224, 86, 253, 0.8)';
                        if (l === 'U2R') return 'rgba(9, 132, 227, 0.8)';
                        return 'rgba(108, 92, 231, 0.8)';
                    }),
                    borderColor: 'rgba(18, 18, 25, 1)',
                    borderWidth: 4,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#f0f0f5', font: { family: 'Outfit', size: 14 } } },
                    title: { display: true, text: 'Probability Distribution', color: '#8a8a9e', font: { family: 'Outfit', size: 16 } }
                },
                cutout: '75%'
            }
        });
    } else {
        chartContainer.style.display = 'none';
    }
}

// ============================================
// CSV UPLOAD & BATCH DETECTION
// ============================================
function handleCSVUpload(input) {
    const file = input.files[0];
    if (!file) return;

    document.getElementById('chart-container').style.display = 'none';

    const resultContainer = document.getElementById('result-container');
    resultContainer.innerHTML = `
        <div class="placeholder-state">
            <div class="radar-scan"></div>
            <h3 style="color: var(--accent);">Processing Batch Telemetry...</h3>
            <p>Analyzing network packets against Gradient Boosting baseline</p>
        </div>
    `;

    const reader = new FileReader();
    reader.onload = async function(e) {
        const text = e.target.result;
        const rows = text.split('\\n').filter(row => row.trim() !== '');
        if (rows.length < 2) {
            alert('CSV file must contain a header row and at least one data row.');
            return;
        }

        const headers = rows[0].split(',').map(h => h.trim());
        const expectedHeaders = ['protocol', 'service', 'flag', 'duration', 'src_bytes', 'dst_bytes', 'count', 'serror_rate', 'dst_host_count'];
        
        for (const eh of expectedHeaders) {
            if (!headers.includes(eh)) {
                alert('CSV missing required header: ' + eh);
                return;
            }
        }

        const batchData = [];
        for (let i = 1; i < rows.length; i++) {
            const values = rows[i].split(',').map(v => v.trim());
            const record = {};
            for (let j = 0; j < headers.length; j++) {
                record[headers[j]] = values[j];
            }
            batchData.push({
                protocol_type: record.protocol || record.protocol_type,
                service: record.service,
                flag: record.flag,
                duration: parseFloat(record.duration) || 0,
                src_bytes: parseFloat(record.src_bytes) || 0,
                dst_bytes: parseFloat(record.dst_bytes) || 0,
                count: parseFloat(record.count) || 0,
                serror_rate: parseFloat(record.serror_rate) || 0,
                dst_host_count: parseFloat(record.dst_host_count) || 0
            });
        }

        try {
            const response = await fetch(`${API_BASE}/predict_batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ records: batchData })
            });

            if (!response.ok) throw new Error('Backend error');

            const result = await response.json();
            const results = result.results;

            let attackCount = 0;
            results.forEach(r => {
                if (r.prediction !== 'normal') attackCount++;
            });

            const attackPercent = Math.round((attackCount / results.length) * 100);
            const statusColor = attackCount > 0 ? 'var(--danger)' : 'var(--success)';
            const statusGlow = attackCount > 0 ? 'var(--danger-glow)' : 'var(--success-glow)';
            
            let html = `
                <div class="result-header" style="border-left: 4px solid ${statusColor}; box-shadow: 0 0 20px ${statusGlow};">
                    <div class="result-title">
                        <span style="font-size: 2.5rem;">📊</span>
                        <div>
                            <h3 style="color: ${statusColor};">Batch Analysis Complete</h3>
                            <p>Scanned ${results.length} connection records</p>
                        </div>
                    </div>
                </div>

                <div class="recommendation-box" style="border-color: ${statusColor}; box-shadow: inset 0 0 20px ${statusGlow};">
                    <div class="rec-label">THREAT DETECTIONS</div>
                    <div class="rec-action" style="color: ${statusColor}">
                        ${attackCount} INTRUSIONS LOGGED (${attackPercent}%)
                    </div>
                </div>
            `;
            resultContainer.innerHTML = html;

        } catch (error) {
            console.error(error);
            alert('Failed to process batch data.');
        }
    };
    reader.readAsText(file);
}

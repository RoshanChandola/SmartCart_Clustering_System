/* ==========================================
   SmartCart Customer Clustering JavaScript
   Client-side Logic & Visualizations
   ========================================== */

document.addEventListener("DOMContentLoaded", () => {
    // State Variables
    let statsData = null;
    let selectedFile = null;
    let processedCSVRaw = null;
    
    // DOM Elements - Navigation
    const navItems = document.querySelectorAll(".nav-item");
    const tabViews = document.querySelectorAll(".tab-view");
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");
    
    // DOM Elements - Loaders
    const globalLoader = document.getElementById("global-loader");
    const loaderText = document.getElementById("loader-text");
    
    // DOM Elements - Overview Metrics
    const metricCustomers = document.getElementById("metric-total-customers");
    const metricIncome = document.getElementById("metric-avg-income");
    const metricSpending = document.getElementById("metric-avg-spending");
    const metricResponse = document.getElementById("metric-response-rate");
    
    // DOM Elements - Segment Explorer
    const segmentCards = document.querySelectorAll(".segment-card");
    const detailIcon = document.getElementById("detail-icon");
    const detailName = document.getElementById("detail-name");
    const detailDesc = document.getElementById("detail-description-short");
    const detailSize = document.getElementById("detail-size-badge");
    const detailIncome = document.getElementById("detail-income");
    const detailSpending = document.getElementById("detail-spending");
    const detailChildren = document.getElementById("detail-children");
    const detailLiving = document.getElementById("detail-living");
    const detailRecency = document.getElementById("detail-recency");
    const detailEducation = document.getElementById("detail-education");
    const detailStrategiesList = document.getElementById("detail-strategies-list");
    
    // DOM Elements - Profiler
    const profilerForm = document.getElementById("profiler-form");
    const profilerPlaceholder = document.getElementById("profiler-placeholder-card");
    const profilerResult = document.getElementById("profiler-result-card");
    const resSegmentIcon = document.getElementById("result-segment-icon");
    const resSegmentName = document.getElementById("result-segment-name");
    const resSegmentDesc = document.getElementById("result-segment-description");
    const resStrategies = document.getElementById("result-strategies");
    
    // DOM Elements - Batch Upload
    const dropzone = document.getElementById("csv-dropzone");
    const fileInput = document.getElementById("csv-file-input");
    const progressContainer = document.getElementById("upload-progress-container");
    const fileNameDisplay = document.getElementById("selected-file-name");
    const fileSizeDisplay = document.getElementById("selected-file-size");
    const processBtn = document.getElementById("process-csv-btn");
    const batchResultsView = document.getElementById("batch-results-view");
    const batchSummaryText = document.getElementById("batch-summary-text");
    const downloadResultsBtn = document.getElementById("download-results-btn");
    const batchTableBody = document.querySelector("#batch-results-table tbody");
    
    // DOM Elements - Retrain Controls
    const retrainBtn = document.getElementById("retrain-pipeline-btn");
    const quickRetrainBtn = document.getElementById("quick-retrain-btn");
    
    // DOM Elements - Mobile Navigation Toggle Elements
    const mobileHamburger = document.getElementById("mobile-hamburger");
    const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
    const sidebarOverlay = document.getElementById("sidebar-overlay");
    const sidebar = document.querySelector(".sidebar");

    // Chart.js Chart Instances
    let distributionChart = null;
    let incomeSpendChart = null;
    let comparisonChart = null;

    // Segment Meta definitions (synchronized with backend names/colors/icons)
    const SEGMENT_META = {
        0: { name: "Partnered Budget Families", icon: "👨‍👩‍👧‍👦", color: "#9b5de5" },
        1: { name: "Partnered VIP Spenders", icon: "💎", color: "#00f5d4" },
        2: { name: "Single Budget Parents", icon: "🛒", color: "#ff007f" },
        3: { name: "Responsive Single VIPs", icon: "🎯", color: "#00bbf9" }
    };

    /* ==========================================
       Utility Functions
       ========================================== */
    const showLoader = (text = "Loading insights...") => {
        loaderText.textContent = text;
        globalLoader.classList.remove("hidden");
    };

    const hideLoader = () => {
        globalLoader.classList.add("hidden");
    };

    const formatCurrency = (val) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            maximumFractionDigits: 0
        }).format(val);
    };

    /* ==========================================
       Mobile Navigation Sidebar Drawer Toggling
       ========================================== */
    const openMobileSidebar = () => {
        if (sidebar && sidebarOverlay) {
            sidebar.classList.add("open");
            sidebarOverlay.classList.add("active");
        }
    };

    const closeMobileSidebar = () => {
        if (sidebar && sidebarOverlay) {
            sidebar.classList.remove("open");
            sidebarOverlay.classList.remove("active");
        }
    };

    if (mobileHamburger) {
        mobileHamburger.addEventListener("click", openMobileSidebar);
    }
    if (sidebarCloseBtn) {
        sidebarCloseBtn.addEventListener("click", closeMobileSidebar);
    }
    if (sidebarOverlay) {
        sidebarOverlay.addEventListener("click", closeMobileSidebar);
    }

    /* ==========================================
       Tab Switching Logic
       ========================================== */
    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const tabName = item.getAttribute("data-tab");
            
            // Toggle active nav class
            navItems.forEach(nav => nav.classList.remove("active"));
            item.classList.add("active");
            
            // Toggle active tab view
            tabViews.forEach(view => view.classList.remove("active"));
            const targetView = document.getElementById(`tab-${tabName}`);
            targetView.classList.add("active");
            
            // Update Page Title
            switch(tabName) {
                case "dashboard":
                    pageTitle.textContent = "Dashboard Overview";
                    pageSubtitle.textContent = "Real-time unsupervised customer behavior segmentation";
                    // Re-render plotly plot to handle size resize issues
                    if (statsData) draw3DScatterPlot(statsData.pca_points);
                    break;
                case "explorer":
                    pageTitle.textContent = "Customer Segment Explorer";
                    pageSubtitle.textContent = "Drill down into customer behavior traits and marketing plans";
                    break;
                case "profiler":
                    pageTitle.textContent = "Customer Profiler";
                    pageSubtitle.textContent = "Predict a single customer segment using active pipeline components";
                    break;
                case "batch":
                    pageTitle.textContent = "Batch Customer Segmenter";
                    pageSubtitle.textContent = "Upload client list in CSV to bulk segment customer base";
                    break;
                case "settings":
                    pageTitle.textContent = "Model & Pipeline Settings";
                    pageSubtitle.textContent = "Inspect machine learning components and retrain options";
                    break;
            }

            // Close mobile sidebar on navigation click
            closeMobileSidebar();
        });
    });

    /* ==========================================
       AJAX API Requests
       ========================================== */
    const loadOverviewData = async (retrain = false) => {
        showLoader(retrain ? "Fitting and retraining ML pipeline. Please wait..." : "Loading customer analytics...");
        try {
            const endpoint = retrain ? "/api/retrain" : "/api/stats";
            const response = await fetch(endpoint, retrain ? { method: "POST" } : { method: "GET" });
            
            if (!response.ok) throw new Error("HTTP error " + response.status);
            
            const data = await response.json();
            
            // If it was a retrain, fetch stats again
            if (retrain) {
                hideLoader();
                loadOverviewData();
                return;
            }
            
            statsData = data;
            
            // 1. Update metric widgets
            metricCustomers.textContent = statsData.overview.total_customers.toLocaleString();
            metricIncome.textContent = formatCurrency(statsData.overview.avg_income);
            metricSpending.textContent = formatCurrency(statsData.overview.avg_spending);
            metricResponse.textContent = statsData.overview.response_rate.toFixed(1) + "%";
            
            // 2. Draw charts
            drawDistributionChart(statsData.clusters);
            drawIncomeSpendChart(statsData.pca_points);
            draw3DScatterPlot(statsData.pca_points);
            
            // 3. Set default in Explorer
            selectSegment(0);
            
            hideLoader();
        } catch (err) {
            console.error("Error loading analytics:", err);
            hideLoader();
            alert("Failed to connect to the backend server. Make sure FastAPI server is running.");
        }
    };

    /* ==========================================
       Chart.js & Plotly Visualizations
       ========================================== */
    const drawDistributionChart = (clusters) => {
        const ctx = document.getElementById("chart-distribution").getContext("2d");
        
        if (distributionChart) distributionChart.destroy();
        
        const labels = [];
        const counts = [];
        const colors = [];
        
        for (let i = 0; i < 4; i++) {
            labels.push(SEGMENT_META[i].name);
            counts.push(clusters[i].size);
            colors.push(SEGMENT_META[i].color);
        }
        
        distributionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: colors,
                    borderWidth: 1,
                    borderColor: 'rgba(255,255,255,0.08)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#a0aec0',
                            padding: 15,
                            font: { family: 'Plus Jakarta Sans', size: 11 }
                        }
                    }
                },
                cutout: '70%'
            }
        });
    };

    const drawIncomeSpendChart = (points) => {
        const ctx = document.getElementById("chart-income-spend").getContext("2d");
        
        if (incomeSpendChart) incomeSpendChart.destroy();
        
        // Sampling points to make chart lightweight
        const sampledPoints = points.filter((_, idx) => idx % 2 === 0);
        
        const datasets = [];
        for (let c = 0; c < 4; c++) {
            datasets.push({
                label: SEGMENT_META[c].name,
                data: sampledPoints.filter(p => p.cluster === c).map(p => ({ x: p.income, y: p.spending })),
                backgroundColor: SEGMENT_META[c].color,
                pointRadius: 3,
                hoverRadius: 5
            });
        }
        
        incomeSpendChart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#a0aec0',
                            font: { family: 'Plus Jakarta Sans', size: 10 }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Income ($)', color: '#a0aec0' },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#a0aec0' }
                    },
                    y: {
                        title: { display: true, text: 'Total Spending ($)', color: '#a0aec0' },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                        ticks: { color: '#a0aec0' }
                    }
                }
            }
        });
    };

    const draw3DScatterPlot = (points) => {
        const traces = [];
        
        for (let c = 0; c < 4; c++) {
            const clusterPoints = points.filter(p => p.cluster === c);
            traces.push({
                x: clusterPoints.map(p => p.x),
                y: clusterPoints.map(p => p.y),
                z: clusterPoints.map(p => p.z),
                mode: 'markers',
                type: 'scatter3d',
                name: SEGMENT_META[c].name,
                marker: {
                    size: 3.5,
                    color: SEGMENT_META[c].color,
                    opacity: 0.85
                },
                text: clusterPoints.map(p => `ID: ${p.id}<br>Income: ${formatCurrency(p.income)}<br>Spending: ${formatCurrency(p.spending)}<br>Age: ${p.age}`),
                hoverinfo: 'text'
            });
        }
        
        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: { title: 'PC1', color: '#a0aec0', gridcolor: 'rgba(255,255,255,0.05)', showbackground: false },
                yaxis: { title: 'PC2', color: '#a0aec0', gridcolor: 'rgba(255,255,255,0.05)', showbackground: false },
                zaxis: { title: 'PC3', color: '#a0aec0', gridcolor: 'rgba(255,255,255,0.05)', showbackground: false },
                backgroundcolor: 'rgba(0,0,0,0)'
            },
            legend: {
                font: { color: '#a0aec0', family: 'Plus Jakarta Sans', size: 10 },
                orientation: 'h',
                yanchor: 'bottom',
                y: -0.05,
                xanchor: 'center',
                x: 0.5
            }
        };
        
        Plotly.newPlot('pca-3d-scatter', traces, layout, {
            responsive: true,
            displayModeBar: false
        });
    };

    /* ==========================================
       Segment Explorer Interaction
       ========================================== */
    const selectSegment = (clusterId) => {
        if (!statsData) return;
        
        segmentCards.forEach(card => card.classList.remove("active"));
        const activeCard = document.querySelector(`.segment-card[data-cluster="${clusterId}"]`);
        if (activeCard) activeCard.classList.add("active");
        
        const cluster = statsData.clusters[clusterId];
        
        // Dynamically grab details mapped from backend info
        detailIcon.textContent = SEGMENT_META[clusterId].icon;
        detailIcon.style.color = SEGMENT_META[clusterId].color;
        detailName.textContent = SEGMENT_META[clusterId].name;
        
        // Get hardcoded texts or retrieve from API
        let description = "";
        let strategies = [];
        
        if (clusterId === 0) {
            description = "Middle-income couples or families with children. They have low-to-moderate spending patterns and high web search activity relative to actual purchases. They are careful spenders who look for deals and discounts.";
            strategies = [
                "Provide family-oriented discounts and multi-buy package deals.",
                "Run price-drop alerts and promo code campaigns to drive web-to-store conversions.",
                "Highlight high-value, cost-effective alternatives in product searches."
            ];
        } else if (clusterId === 1) {
            description = "High-income couples or families without children. They spend very heavily across all categories, especially on premium products, and prefer shopping directly through store and catalog channels. They are not very price-sensitive.";
            strategies = [
                "Promote premium product catalogs (e.g., fine wines, high-end meats).",
                "Offer a VIP loyalty program, exclusive customer service lines, and store reservation perks.",
                "Create high-value product bundle recommendations for couples."
            ];
        } else if (clusterId === 2) {
            description = "Lower-income single parents. They have the lowest average spending and highest children ratio. They visit the online store very frequently but rarely complete high-ticket transactions. They are highly price-conscious.";
            strategies = [
                "Deploy highly target-specific budget deals and clearance campaigns.",
                "Promote low-cost food, kids items, and high-frequency basic commodities.",
                "Offer flexible payment terms, loyalty points, or free shipping threshold updates."
            ];
        } else if (clusterId === 3) {
            description = "High-income singles without children. They are big spenders, shop frequently online and in-store, and are extremely responsive to marketing campaigns (32% acceptance rate—highest across all segments). They are the primary targets for promotions.";
            strategies = [
                "Target with new product launches and marketing campaigns immediately (high conversion rate).",
                "Promote luxury items, gourmet food, gold/specialty goods, and single-serve convenience products.",
                "Use email marketing and personalized web recommendations for quick conversions."
            ];
        }
        
        detailDesc.textContent = description;
        detailSize.textContent = `${cluster.size.toLocaleString()} Customers (${cluster.percentage.toFixed(1)}%)`;
        detailSize.style.backgroundColor = `${SEGMENT_META[clusterId].color}25`;
        detailSize.style.color = SEGMENT_META[clusterId].color;
        detailSize.style.borderColor = `${SEGMENT_META[clusterId].color}50`;
        
        detailIncome.textContent = formatCurrency(cluster.Income);
        detailSpending.textContent = formatCurrency(cluster.Total_Spending);
        detailChildren.textContent = `~${cluster.Total_Children.toFixed(2)} children`;
        detailLiving.textContent = cluster.Living_With;
        detailRecency.textContent = `${cluster.Recency.toFixed(1)} days ago`;
        detailEducation.textContent = cluster.Education;
        
        // Populate Strategies
        detailStrategiesList.innerHTML = "";
        strategies.forEach(strat => {
            const li = document.createElement("li");
            li.textContent = strat;
            detailStrategiesList.appendChild(li);
        });
    };

    segmentCards.forEach(card => {
        card.addEventListener("click", () => {
            const clusterId = parseInt(card.getAttribute("data-cluster"));
            selectSegment(clusterId);
        });
    });

    /* ==========================================
       Customer Profiler Logic
       ========================================== */
    profilerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        showLoader("Running classifier...");
        
        const payload = {
            Year_Birth: parseInt(document.getElementById("p-year-birth").value),
            Income: parseFloat(document.getElementById("p-income").value),
            Education: document.getElementById("p-education").value,
            Marital_Status: document.getElementById("p-marital").value,
            Kidhome: parseInt(document.getElementById("p-kidhome").value),
            Teenhome: parseInt(document.getElementById("p-teenhome").value),
            Dt_Customer: document.getElementById("p-dt-customer").value,
            Recency: parseInt(document.getElementById("p-recency").value),
            MntWines: parseFloat(document.getElementById("p-wines").value) || 0.0,
            MntFruits: parseFloat(document.getElementById("p-fruits").value) || 0.0,
            MntMeatProducts: parseFloat(document.getElementById("p-meats").value) || 0.0,
            MntFishProducts: parseFloat(document.getElementById("p-fish").value) || 0.0,
            MntSweetProducts: parseFloat(document.getElementById("p-sweets").value) || 0.0,
            MntGoldProds: parseFloat(document.getElementById("p-gold").value) || 0.0,
            NumDealsPurchases: parseInt(document.getElementById("p-deals-purch").value) || 0,
            NumWebPurchases: parseInt(document.getElementById("p-web-purch").value) || 0,
            NumCatalogPurchases: parseInt(document.getElementById("p-catalog-purch").value) || 0,
            NumStorePurchases: parseInt(document.getElementById("p-store-purch").value) || 0,
            NumWebVisitsMonth: parseInt(document.getElementById("p-web-visits").value) || 0,
            Complain: document.getElementById("p-complain").checked ? 1 : 0,
            Response: document.getElementById("p-response").checked ? 1 : 0
        };
        
        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) throw new Error("API classification error");
            
            const data = await response.json();
            
            // Hide placeholder, display result card
            profilerPlaceholder.classList.add("hidden");
            profilerResult.classList.remove("hidden");
            
            // Update Text Details
            resSegmentIcon.textContent = data.cluster_details.icon;
            resSegmentIcon.style.color = data.cluster_details.color;
            resSegmentName.textContent = data.cluster_details.name;
            resSegmentDesc.textContent = data.cluster_details.description;
            
            // Update recommended strategies
            resStrategies.innerHTML = "";
            data.cluster_details.strategies.forEach(strat => {
                const li = document.createElement("li");
                li.textContent = strat;
                resStrategies.appendChild(li);
            });
            
            // Draw comparative radar / bar chart
            drawComparisonChart(data.comparison);
            
            hideLoader();
            
            // Scroll to results card on mobile
            profilerResult.scrollIntoView({ behavior: 'smooth' });
        } catch (err) {
            console.error("Prediction error:", err);
            hideLoader();
            alert("Failed to process customer profile classification.");
        }
    });

    const drawComparisonChart = (comparison) => {
        const ctx = document.getElementById("chart-comparison-radar").getContext("2d");
        
        if (comparisonChart) comparisonChart.destroy();
        
        const labels = ['Income ($)', 'Total Spending ($)', 'Age', 'Total Children', 'Recency (Days)'];
        const custData = [
            comparison.customer.Income,
            comparison.customer.Total_Spending,
            comparison.customer.Age,
            comparison.customer.Total_Children,
            comparison.customer.Recency
        ];
        const avgData = [
            comparison.cluster_averages.Income,
            comparison.cluster_averages.Total_Spending,
            comparison.cluster_averages.Age,
            comparison.cluster_averages.Total_Children,
            comparison.cluster_averages.Recency
        ];
        
        // We will normalize the values relative to the cluster averages to make it plot nicely on a radar chart
        // Formula: (Customer / Cluster Average) * 100
        const normalizedCust = custData.map((val, idx) => {
            const avg = avgData[idx];
            if (avg === 0) return 0;
            return (val / avg) * 100;
        });
        const normalizedAvg = [100, 100, 100, 100, 100]; // Averaged indexes represent 100% baseline
        
        comparisonChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'This Customer (%)',
                        data: normalizedCust,
                        backgroundColor: 'rgba(155, 93, 229, 0.2)',
                        borderColor: '#9b5de5',
                        borderWidth: 2,
                        pointBackgroundColor: '#9b5de5'
                    },
                    {
                        label: 'Cluster Average Benchmark (100%)',
                        data: normalizedAvg,
                        backgroundColor: 'rgba(255,255,255,0.03)',
                        borderColor: 'rgba(255,255,255,0.3)',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointBackgroundColor: 'rgba(255,255,255,0.5)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#a0aec0', font: { family: 'Plus Jakarta Sans', size: 10 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const idx = context.dataIndex;
                                const isCust = context.datasetIndex === 0;
                                const originalVal = isCust ? custData[idx] : avgData[idx];
                                return `${context.dataset.label}: ${idx === 0 || idx === 1 ? formatCurrency(originalVal) : originalVal.toFixed(1)}`;
                            }
                        }
                    }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        pointLabels: { color: '#a0aec0', font: { family: 'Plus Jakarta Sans', size: 11 } },
                        ticks: { color: 'rgba(255, 255, 255, 0.3)', backdropColor: 'transparent', showLabelBackdrop: false }
                    }
                }
            }
        });
    };

    /* ==========================================
       Batch Upload Logic
       ========================================== */
    // Trigger file dialog
    dropzone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag-and-drop event handlers
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--purple)";
        dropzone.style.background = "rgba(155, 93, 229, 0.05)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "rgba(155, 93, 229, 0.3)";
        dropzone.style.background = "rgba(155, 93, 229, 0.02)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(155, 93, 229, 0.3)";
        dropzone.style.background = "rgba(155, 93, 229, 0.02)";
        
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    const handleSelectedFile = (file) => {
        if (!file.name.endsWith(".csv")) {
            alert("Unsupported file format. Please upload a CSV file.");
            return;
        }
        
        selectedFile = file;
        fileNameDisplay.innerHTML = `<i class="fa-solid fa-file-csv"></i> ${file.name}`;
        
        // Formulate readable file size
        const sizeInKB = file.size / 1024;
        fileSizeDisplay.textContent = sizeInKB > 1024 
            ? (sizeInKB / 1024).toFixed(1) + " MB" 
            : sizeInKB.toFixed(0) + " KB";
            
        progressContainer.classList.remove("hidden");
    };

    processBtn.addEventListener("click", async () => {
        if (!selectedFile) return;
        
        showLoader("Uploading and processing batch CSV data...");
        
        const formData = new FormData();
        formData.append("file", selectedFile);
        
        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server failed to process upload");
            }
            
            const data = await response.json();
            
            processedCSVRaw = data.full_csv_raw;
            
            // 1. Show batch summary view
            batchResultsView.classList.remove("hidden");
            batchSummaryText.textContent = `Successfully processed ${data.total_records.toLocaleString()} customer records`;
            
            // 2. Update cluster count breakdown
            document.getElementById("batch-count-0").textContent = data.counts["0"].toLocaleString();
            document.getElementById("batch-count-1").textContent = data.counts["1"].toLocaleString();
            document.getElementById("batch-count-2").textContent = data.counts["2"].toLocaleString();
            document.getElementById("batch-count-3").textContent = data.counts["3"].toLocaleString();
            
            // 3. Populate Table Preview (top 500 rows)
            batchTableBody.innerHTML = "";
            data.preview.forEach(row => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><code>#${row.ID}</code></td>
                    <td>${row.Age}</td>
                    <td>${row.Income ? formatCurrency(row.Income) : 'N/A'}</td>
                    <td>${row.Education}</td>
                    <td>${row.Living_With} (${row.Total_Children} Kids)</td>
                    <td>${row.Recency} days</td>
                    <td><span class="badge" style="background-color: ${SEGMENT_META[row.Cluster].color}25; color: ${SEGMENT_META[row.Cluster].color}; border-color: ${SEGMENT_META[row.Cluster].color}50">${row.Cluster_Name}</span></td>
                `;
                batchTableBody.appendChild(tr);
            });
            
            hideLoader();
            
            // Scroll to results table
            batchResultsView.scrollIntoView({ behavior: 'smooth' });
        } catch (err) {
            console.error("Batch processing error:", err);
            hideLoader();
            alert(err.message || "Failed to process bulk upload. Verify that the file layout conforms to model standards.");
        }
    });

    // Handle CSV Download
    downloadResultsBtn.addEventListener("click", () => {
        if (!processedCSVRaw) return;
        
        const blob = new Blob([processedCSVRaw], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        
        a.setAttribute('href', url);
        a.setAttribute('download', `segmented_${selectedFile ? selectedFile.name : 'customers.csv'}`);
        a.click();
        
        window.URL.revokeObjectURL(url);
    });

    /* ==========================================
       Retraining Triggers
       ========================================== */
    const triggerRetrain = () => {
        const confirmRetrain = confirm("Are you sure you want to retrain the ML pipeline models? This runs the clustering algorithm on data/smartcart_customers.csv and updates the saved weights.");
        if (confirmRetrain) {
            loadOverviewData(true);
        }
    };

    retrainBtn.addEventListener("click", triggerRetrain);
    quickRetrainBtn.addEventListener("click", triggerRetrain);

    /* ==========================================
       Initialize App
       ========================================== */
    loadOverviewData();
});

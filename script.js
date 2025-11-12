// ============================================
// CS180 Projects - Main JavaScript File
// ============================================

// Load Prism.js for syntax highlighting
(function() {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css';
    document.head.appendChild(link);
    
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js';
    script.defer = true;
    document.head.appendChild(script);
})();

// Dark Mode Toggle
(function() {
    const themeToggle = document.querySelector('.theme-toggle');
    const html = document.documentElement;
    
    // Get saved theme or default to system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
        html.setAttribute('data-theme', savedTheme);
    } else if (prefersDark) {
        html.setAttribute('data-theme', 'dark');
    }
    
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeIcon(themeToggle, newTheme);
        });
        
        updateThemeIcon(themeToggle, html.getAttribute('data-theme'));
    }
    
    function updateThemeIcon(button, theme) {
        button.textContent = theme === 'dark' ? '☀️' : '🌙';
        button.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    }
})();

// Progress Bar
(function() {
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', () => {
        const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (window.scrollY / windowHeight) * 100;
        progressBar.style.width = scrolled + '%';
    });
})();

// Image Lightbox
(function() {
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = `
        <span class="lightbox-close">&times;</span>
        <span class="lightbox-prev">&#10094;</span>
        <span class="lightbox-next">&#10095;</span>
        <div class="lightbox-content">
            <img class="lightbox-image" src="" alt="">
            <div class="lightbox-caption"></div>
        </div>
    `;
    document.body.appendChild(lightbox);
    
    let currentImages = [];
    let currentIndex = 0;
    
    // Initialize lightbox for all gallery images
    function initLightbox() {
        const galleryImages = document.querySelectorAll('.image-gallery img, .image-container img, .section img');
        galleryImages.forEach((img, index) => {
            if (!img.closest('.image-comparison')) {
                img.style.cursor = 'pointer';
                img.addEventListener('click', () => {
                    openLightbox(index, Array.from(galleryImages));
                });
            }
        });
    }
    
    function openLightbox(index, images) {
        currentImages = images;
        currentIndex = index;
        updateLightbox();
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
    }
    
    function updateLightbox() {
        if (currentImages.length === 0) return;
        const img = currentImages[currentIndex];
        const lightboxImg = lightbox.querySelector('.lightbox-image');
        const caption = lightbox.querySelector('.lightbox-caption');
        
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt;
        caption.textContent = img.alt || '';
        
        // Update navigation visibility
        lightbox.querySelector('.lightbox-prev').style.display = currentIndex > 0 ? 'block' : 'none';
        lightbox.querySelector('.lightbox-next').style.display = currentIndex < currentImages.length - 1 ? 'block' : 'none';
    }
    
    function nextImage() {
        if (currentIndex < currentImages.length - 1) {
            currentIndex++;
            updateLightbox();
        }
    }
    
    function prevImage() {
        if (currentIndex > 0) {
            currentIndex--;
            updateLightbox();
        }
    }
    
    lightbox.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
    lightbox.querySelector('.lightbox-prev').addEventListener('click', prevImage);
    lightbox.querySelector('.lightbox-next').addEventListener('click', nextImage);
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) closeLightbox();
    });
    
    document.addEventListener('keydown', (e) => {
        if (!lightbox.classList.contains('active')) return;
        if (e.key === 'Escape') closeLightbox();
        if (e.key === 'ArrowLeft') prevImage();
        if (e.key === 'ArrowRight') nextImage();
    });
    
    initLightbox();
})();

// Table of Contents
(function() {
    function generateTOC() {
        const main = document.querySelector('main');
        if (!main) return;
        
        const sections = main.querySelectorAll('section h2, .section h2');
        if (sections.length < 2) return; // Only show TOC if there are multiple sections
        
        const toc = document.createElement('div');
        toc.className = 'table-of-contents';
        const tocTitle = document.createElement('h3');
        tocTitle.textContent = 'Table of Contents';
        toc.appendChild(tocTitle);
        
        const tocList = document.createElement('ul');
        sections.forEach((heading, index) => {
            const id = `section-${index}`;
            heading.id = id;
            
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `#${id}`;
            a.textContent = heading.textContent;
            a.addEventListener('click', (e) => {
                e.preventDefault();
                heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
            li.appendChild(a);
            tocList.appendChild(li);
        });
        
        toc.appendChild(tocList);
        
        // Insert TOC at the beginning of main
        const firstSection = main.querySelector('section, .section');
        if (firstSection) {
            firstSection.parentNode.insertBefore(toc, firstSection);
        }
        
        // Highlight active section
        const tocLinks = toc.querySelectorAll('a');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    tocLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, { rootMargin: '-20% 0px -70% 0px' });
        
        sections.forEach(section => observer.observe(section));
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', generateTOC);
    } else {
        generateTOC();
    }
})();

// Back to Top Button
(function() {
    const backToTop = document.createElement('button');
    backToTop.className = 'back-to-top';
    backToTop.innerHTML = '↑';
    backToTop.setAttribute('aria-label', 'Back to top');
    document.body.appendChild(backToTop);
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });
    
    backToTop.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
})();

// Search Functionality
(function() {
    const searchContainer = document.createElement('div');
    searchContainer.className = 'search-container';
    searchContainer.innerHTML = `
        <input type="text" class="search-input" placeholder="Search projects..." aria-label="Search">
        <div class="search-results"></div>
    `;
    
    const header = document.querySelector('header');
    if (header) {
        header.appendChild(searchContainer);
    }
    
    const searchInput = searchContainer.querySelector('.search-input');
    const searchResults = searchContainer.querySelector('.search-results');
    
    // Simple search index (could be expanded)
    const searchData = [];
    
    function buildSearchIndex() {
        const projects = document.querySelectorAll('.project-link');
        projects.forEach(project => {
            const title = project.querySelector('h3')?.textContent || '';
            const description = project.querySelector('p')?.textContent || '';
            const link = project.getAttribute('href') || '';
            searchData.push({ title, description, link });
        });
    }
    
    function performSearch(query) {
        if (!query.trim()) {
            searchResults.classList.remove('active');
            return;
        }
        
        const results = searchData.filter(item => {
            const searchText = `${item.title} ${item.description}`.toLowerCase();
            return searchText.includes(query.toLowerCase());
        });
        
        displayResults(results);
    }
    
    function displayResults(results) {
        searchResults.innerHTML = '';
        
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
            searchResults.classList.add('active');
            return;
        }
        
        results.forEach(result => {
            const item = document.createElement('div');
            item.className = 'search-result-item';
            item.innerHTML = `
                <div class="search-result-title">${result.title}</div>
                <div class="search-result-snippet">${result.description}</div>
            `;
            item.addEventListener('click', () => {
                window.location.href = result.link;
            });
            searchResults.appendChild(item);
        });
        
        searchResults.classList.add('active');
    }
    
    searchInput.addEventListener('input', (e) => {
        performSearch(e.target.value);
    });
    
    document.addEventListener('click', (e) => {
        if (!searchContainer.contains(e.target)) {
            searchResults.classList.remove('active');
        }
    });
    
    buildSearchIndex();
})();

// Image Comparison Slider
(function() {
    function initImageComparison(container) {
        const beforeImg = container.querySelector('.image-comparison-before img');
        const afterImg = container.querySelector('.image-comparison-after img');
        const slider = container.querySelector('.image-comparison-slider');
        
        if (!beforeImg || !afterImg || !slider) return;
        
        let isDragging = false;
        
        function updateSlider(x) {
            const rect = container.getBoundingClientRect();
            const percentage = ((x - rect.left) / rect.width) * 100;
            const clampedPercentage = Math.max(0, Math.min(100, percentage));
            
            slider.style.left = clampedPercentage + '%';
            afterImg.style.clipPath = `inset(0 ${100 - clampedPercentage}% 0 0)`;
        }
        
        slider.addEventListener('mousedown', () => {
            isDragging = true;
        });
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                updateSlider(e.clientX);
            }
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        container.addEventListener('click', (e) => {
            if (!isDragging) {
                updateSlider(e.clientX);
            }
        });
        
        // Touch support
        slider.addEventListener('touchstart', (e) => {
            e.preventDefault();
            isDragging = true;
        });
        
        document.addEventListener('touchmove', (e) => {
            if (isDragging && e.touches.length > 0) {
                updateSlider(e.touches[0].clientX);
            }
        });
        
        document.addEventListener('touchend', () => {
            isDragging = false;
        });
    }
    
    document.querySelectorAll('.image-comparison').forEach(initImageComparison);
})();

// Lazy Loading Enhancement
(function() {
    if ('loading' in HTMLImageElement.prototype) {
        // Native lazy loading supported
        return;
    }
    
    // Fallback for browsers without native lazy loading
    const images = document.querySelectorAll('img[loading="lazy"]');
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                }
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => {
        if (img.dataset.src) {
            imageObserver.observe(img);
        }
    });
})();

// Add loading attribute to all images
document.addEventListener('DOMContentLoaded', () => {
    const images = document.querySelectorAll('img:not([loading])');
    images.forEach(img => {
        img.setAttribute('loading', 'lazy');
    });
});


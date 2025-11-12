// ============================================
// Portfolio - Main JavaScript File
// ============================================

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


// Image Lightbox - Simplified (open/close only)
(function() {
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = `
        <span class="lightbox-close">&times;</span>
        <div class="lightbox-content">
            <img class="lightbox-image" src="" alt="">
            <div class="lightbox-caption"></div>
        </div>
    `;
    document.body.appendChild(lightbox);
    
    // Initialize lightbox for all gallery images
    function initLightbox() {
        const galleryImages = document.querySelectorAll('.image-gallery img, .image-container img, .section img');
        galleryImages.forEach((img) => {
            img.style.cursor = 'pointer';
            img.addEventListener('click', () => {
                openLightbox(img);
            });
        });
    }
    
    function openLightbox(img) {
        const lightboxImg = lightbox.querySelector('.lightbox-image');
        const caption = lightbox.querySelector('.lightbox-caption');
        
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt;
        caption.textContent = img.alt || '';
        
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
    }
    
    lightbox.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) closeLightbox();
    });
    
    document.addEventListener('keydown', (e) => {
        if (!lightbox.classList.contains('active')) return;
        if (e.key === 'Escape') closeLightbox();
    });
    
    initLightbox();
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

// Side Navigation for Project Pages
(function() {
    function generateSideNav() {
        const main = document.querySelector('main');
        if (!main) return;
        
        // Only generate on project pages (not homepage)
        const isHomepage = main.querySelector('.project-links');
        if (isHomepage) return;
        
        const sections = main.querySelectorAll('section h2, .section h2');
        if (sections.length < 2) return; // Only show nav if there are multiple sections
        
        // Create side nav
        const sideNav = document.createElement('nav');
        sideNav.className = 'side-nav';
        sideNav.setAttribute('aria-label', 'Project navigation');
        
        const navTitle = document.createElement('h3');
        navTitle.textContent = 'Contents';
        sideNav.appendChild(navTitle);
        
        const navList = document.createElement('ul');
        
        sections.forEach((heading) => {
            // Generate ID from heading text
            const id = heading.textContent
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, '-')
                .replace(/^-|-$/g, '');
            heading.id = id;
            
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `#${id}`;
            a.textContent = heading.textContent;
            
            a.addEventListener('click', (e) => {
                e.preventDefault();
                const headingRect = heading.getBoundingClientRect();
                const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
                const targetPosition = currentScroll + headingRect.top - 40;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
                
                // Update URL without jumping
                history.pushState(null, '', `#${id}`);
            });
            
            li.appendChild(a);
            navList.appendChild(li);
        });
        
        sideNav.appendChild(navList);
        
        // Wrap main content
        const mainContent = document.createElement('div');
        mainContent.className = 'main-content';
        
        // Store all element children first (filter out text nodes)
        const children = Array.from(main.children);
        
        // Move all children of main into mainContent
        children.forEach(child => {
            mainContent.appendChild(child);
        });
        
        // Add side nav and main content to main
        main.appendChild(sideNav);
        main.appendChild(mainContent);
        main.classList.add('has-side-nav');
        
        // Highlight active section on scroll
        const navLinks = sideNav.querySelectorAll('a');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, { 
            rootMargin: '-20% 0px -70% 0px',
            threshold: 0
        });
        
        sections.forEach(section => observer.observe(section));
        
        // Handle initial hash in URL
        if (window.location.hash) {
            const targetId = window.location.hash.substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                setTimeout(() => {
                    const targetRect = target.getBoundingClientRect();
                    const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
                    const targetPosition = currentScroll + targetRect.top - 40;
                    
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }, 100);
            }
        }
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', generateSideNav);
    } else {
        generateSideNav();
    }
})();


# Before & After Examples

Specific examples of what to change, showing exact before/after text.

---

## **MAIN SITE (`index.html`)**

### Title Tag
**Before:**
```html
<title>CS180 Projects</title>
```
**After:**
```html
<title>Computer Vision Portfolio - Adriel Vijuan</title>
```
or
```html
<title>Adriel Vijuan - Computer Vision Projects</title>
```

---

### Main Header
**Before:**
```html
<h1>CS180: Projects</h1>
```
**After:**
```html
<h1>Computer Vision Portfolio</h1>
```
or
```html
<h1>Adriel Vijuan</h1>
<h2>Computer Vision Projects</h2>
```

---

### Project Links
**Before:**
```html
<h3>Project #1</h3>
<p>Colorizing the Prokudin-Gorskii Photo Collection</p>
```
**After:**
```html
<h3>Historical Photo Colorization</h3>
<p>Automated alignment and colorization of Prokudin-Gorskii photographs</p>
```

**Before:**
```html
<h3>Final Project (pt.1)</h3>
<p>HDR Images</p>
```
**After:**
```html
<h3>High Dynamic Range Imaging</h3>
<p>HDR reconstruction and tone mapping from multiple exposures</p>
```

---

### Footer
**Before:**
```html
<footer>
    <p>Created by Adriel Vijuan for CS180 - 2024</p>
</footer>
```
**After:**
```html
<footer>
    <p>© 2024 Adriel Vijuan</p>
</footer>
```
or remove footer entirely

---

## **PROJECT PAGES**

### Page Title & Header
**Before:**
```html
<title>CS180 Project #1</title>
...
<h1>CS180: Project #1</h1>
<h3><a href="../../../index.html">by Adriel Vijuan</a></h3>
<p>Colorizing the <a href="...">Prokudin-Gorskii Photo Collection</a>.</p>
```
**After:**
```html
<title>Historical Photo Colorization - Adriel Vijuan</title>
...
<h1>Historical Photo Colorization</h1>
<p class="author">by <a href="../../../index.html">Adriel Vijuan</a></p>
<p>Automated colorization of the <a href="...">Prokudin-Gorskii Photo Collection</a></p>
```

---

### Section Headers
**Before:**
```html
<h2>Project Overview</h2>
<h2>Approach Breakdown</h2>
<h2>Bells and Whistles</h2>
<h2>Commentary</h2>
```
**After:**
```html
<h2>Overview</h2>
<h2>Methodology</h2>
<h2>Additional Features</h2>
<h2>Discussion</h2>
```

---

### Introduction Text
**Before:**
```html
<p>This project focuses on colorizing the Prokudin-Gorskii photo collection by aligning the color channels of historical 
    black-and-white images. The original black-and-white images consist of film strips of three images, each taken of the same scene, but through a red, blue, 
    and green filter respectively.
    The approach I underwent combines these images into a colorized version and involves several key steps that will be broken down.</p>
```
**After:**
```html
<p>Historical black-and-white photographs often contain valuable color information encoded in separate RGB channel exposures. 
    This work implements an automated pipeline for aligning and combining these channels to reconstruct full-color images from 
    the Prokudin-Gorskii collection, using multi-scale alignment techniques to handle images of varying sizes.</p>
```

---

### Academic Language
**Before:**
```html
<p>Results of the colorization process from the provided data folder for this project:</p>
```
**After:**
```html
<p>Results from the dataset:</p>
```

**Before:**
```html
<p>To establish correspondence points between the two images, 
    I used the provided Correspondence tool to interactively select 
    key facial features. I exported these points as a JSON file and loaded 
    them into my code...</p>
```
**After:**
```html
<p>To establish correspondence points between the two images, 
    I implemented an interactive point selection system that allows 
    precise mapping of key facial features. These correspondence points 
    were exported as JSON and integrated into the morphing pipeline...</p>
```

---

### Personal Anecdotes (Remove)
**Before:**
```html
<p>As someone who utilizes Photoshop in my freelance work, exploring these 
    similar types of tools from a fundamental level was incredibly 
    interesting and rewarding for me. Understanding how these image 
    manipulation techniques work under the hood enhances my appreciation 
    for the software I use daily...</p>
```
**After:**
```html
<p>This implementation demonstrates the fundamental techniques behind 
    professional image manipulation tools, providing insight into how 
    morphing algorithms work at a computational level.</p>
```
or simply remove entirely

---

### Configuration Sections
**Before:**
```html
<h3>Configuration</h3>
<p>The configuration section of the code determines how image processing is handled based on user input:</p>
<ul>
    <li><strong><code>PROCESS_SINGLE_IMAGE</code></strong>: This variable allows the user to specify whether to process a single image or multiple images within a 
        directory...</li>
</ul>
```
**After:**
```html
<h3>Implementation Details</h3>
<p>The implementation supports both single-image and batch processing modes:</p>
<ul>
    <li><strong><code>PROCESS_SINGLE_IMAGE</code></strong>: Controls whether to process a single image or all images in a directory...</li>
</ul>
```

---

### Step-by-Step Language
**Before:**
```html
<h2>Step-by-Step Process for Part A</h2>
<section class="section">
    <h3>1. Correspondence Point Selection</h3>
    <p>The first step in panorama creation is to identify corresponding points...</p>
</section>
```
**After:**
```html
<h2>Methodology</h2>
<section class="section">
    <h3>Correspondence Point Selection</h3>
    <p>Panorama creation begins by identifying corresponding points...</p>
</section>
```

---

### Image Captions
**Before:**
```html
<p class="description">Displacement vectors for cathedral.jpg: Green channel: [-5, 0], Red channel: [-5, 1]</p>
```
**After:**
```html
<p class="description">Colorized cathedral image</p>
<p class="technical-note">Alignment offsets: G[-5,0], R[-5,1]</p>
```
or
```html
<p class="description">Colorized cathedral image <span class="technical-detail">(G[-5,0], R[-5,1])</span></p>
```

---

### "Part A" / "Part B" Structure
**Before:**
```html
<h2>Part A: Image Warping and Mosaicing</h2>
<h2>Part B: Feature Detection</h2>
```
**After:**
```html
<h2>Image Warping and Mosaicing</h2>
<h2>Automated Feature Detection</h2>
```

---

### Project Goals Section
**Before:**
```html
<h2>Project Goals</h2>
<p>The primary goal of this project is to develop a robust HDR imaging pipeline...</p>
```
**After:**
```html
<h2>Objectives</h2>
<p>This work implements a robust HDR imaging pipeline...</p>
```
or merge into Introduction:
```html
<h2>Introduction</h2>
<p>This work implements a robust HDR imaging pipeline that effectively combines 
    multiple exposures to capture the full dynamic range of a scene. The implementation 
    includes recovering camera response curves, constructing HDR radiance maps, 
    and applying tone mapping operators for display.</p>
```

---

### Code Comments
**Before:**
```javascript
// ============================================
// CS180 Projects - Main JavaScript File
// ============================================
```
**After:**
```javascript
// ============================================
// Portfolio - Main JavaScript File
// ============================================
```
or
```javascript
// Main JavaScript File
```

---

### CSS Comments
**Before:**
```css
/* ============================================
   CS180 Projects - Consolidated Stylesheet
   ============================================ */
```
**After:**
```css
/* ============================================
   Portfolio Stylesheet
   ============================================ */
```
or remove entirely

---

## **META TAGS (Add to All Pages)**

**Add to `<head>` section:**
```html
<meta name="description" content="Automated colorization of historical Prokudin-Gorskii photographs using multi-scale alignment techniques.">
<meta name="keywords" content="computer vision, image processing, colorization, Prokudin-Gorskii, Python">
<meta property="og:title" content="Historical Photo Colorization - Adriel Vijuan">
<meta property="og:description" content="Automated colorization of historical Prokudin-Gorskii photographs using multi-scale alignment techniques.">
<meta property="og:type" content="website">
<meta property="og:image" content="path/to/preview-image.jpg">
```

---

## **AUTHOR ATTRIBUTION**

**Before (Too Prominent):**
```html
<h3><a href="../../../index.html">by Adriel Vijuan</a></h3>
```

**After Option 1 (Subtle):**
```html
<p class="author-link"><a href="../../../index.html">Adriel Vijuan</a></p>
```
with CSS:
```css
.author-link {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}
```

**After Option 2 (Footer):**
```html
<!-- In header, remove author line -->
<!-- In footer: -->
<footer>
    <p><a href="../../../index.html">← Back to Portfolio</a> | © 2024 Adriel Vijuan</p>
</footer>
```

**After Option 3 (Remove if name is in main site header):**
Just remove entirely if your name appears prominently on the main portfolio page.

---

## **SUMMARY OF KEY CHANGES**

1. **Remove course codes:** "CS180" → [delete]
2. **Descriptive names:** "Project #1" → "Historical Photo Colorization"
3. **Professional language:** "provided data" → "dataset"
4. **Remove academic structure:** "Part A" → descriptive names
5. **Subtle attribution:** Prominent `<h3>` → subtle text or footer
6. **Remove personal anecdotes:** Keep it technical
7. **Better captions:** Technical details secondary to descriptions
8. **Professional sections:** "Bells and Whistles" → "Additional Features"

---

These examples show the exact transformations needed. Use find-and-replace where possible, but review each change to ensure it makes sense in context.


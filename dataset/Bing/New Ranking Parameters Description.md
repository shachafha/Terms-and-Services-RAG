Bing Webmaster Guidelines
=========================

These guidelines are intended to help you understand how Bing finds, indexes, and ranks websites.

Following these guidelines will help with indexing of your site on Bing. It will also help you optimize your site to increase its opportunity to rank for relevant queries in Bing’s search results. Please pay special attention to the guidance in the “Abuse section” and examples in the “Things to avoid” section. Following the guidelines will ensure your site plays by the rules and is not found to be spammy, which could lead to demotion or even delisting your website from Bing search results.

How Bing finds and indexes your site
====================================

Help Bing find all your pages
-----------------------------

* **Sitemaps:** Sitemaps are an essential way for Bing to discover URLs and content for your website. It is a file that provides information about the URLs, other files, and content such as images and videos on your website. The sitemap informs the crawler about pages and files you believe are important within your site. It also provides additional information, such as when the page was last updated. We strongly recommend using an [XML sitemap file](https://www.sitemaps.org/) to help Bing discover all the relevant URLs and content within your website. Please keep your sitemap files as up-to-date as possible; refreshing them in real-time or at least once a day. This will enable the timely removal of old URLs and deadlinks once the content is removed from your website.  
      
    * Make your sitemap available to Bing by:
        
        * Submitting it to Bing using the [Bing Webmaster Tools Sitemap tool](https://www.bing.com/webmasters/help/Sitemaps-3b5cf6ed)
        * Inserting the following line in the robots.txt file, specifying the path to your sitemap:  
            Sitemap: [http://example.com/sitemap\_location.xml](http://example.com/sitemap_location.xml)
        
        Once Bing knows your sitemap, Bing will crawl it regularly. There is no need to submit it again except in case of significant changes on the site.  
          
        
    * General sitemap guidelines:
        * Bing supports several sitemap formats including XML, RSS, MRSS, Atom 1.0, and a text file.
        * Use consistent URLs. Bing will only crawl the URLs precisely as listed.
        * Please only list [canonical](https://en.wikipedia.org/wiki/Canonical_link_element#Implementation) URLs in your sitemaps.
        * If your website has multiple versions (HTTP vs HTTPS, or mobile vs desktop), we recommend only pointing to a single version in your sitemap. If you decide to have a unique URL experience for mobile vs desktop, please annotate with the rel\="alternate" attribute.
        * If you have multiple pages for different languages or regions, please use the hreflang tags in either the sitemap or the HTML tag to identify the alternate URLs.
        * Use the <lastmod> attribute to indicate the date and time when the content was last modified.
        * The maximum sitemap size is 50,000 URLs/50MB uncompressed. If your site is large, consider breaking up large sitemaps into smaller sitemaps and, then use a sitemap index file to list all individual sitemaps.
        * Refer to your sitemap in the robots.txt file.
        * If you have not changed your sitemap since Bing crawled it there is no need or benefit to resubmitting it.  
              
            
    * Using a sitemap does not guarantee that all items within a sitemap will be crawled and indexed; however, in most cases you will benefit by having a sitemap as it provides a recommendation and guidance to the crawler.  
          
        
* **Use** [**IndexNow API**](https://www.bing.com/indexnow) or the [**Bing URL or Content Submission API**](https://www.bing.com/webmasters/url-submission-api) to instantly reflect website changes. If you are unable to adopt the APIs, we recommend submitting the updated URLs directly through [Bing Webmaster Tools](https://www.bing.com/webmasters/help/how-to-submit-sitemaps-82a15bd4) or by including them in your sitemap.  
      
    
* **Links**: Links are traditionally regarded as a signal for determining website popularity. The best way to get other sites to link to your site is to create unique and high quality and unique content. Bing’s crawler (Bingbot) follows links within your website (internal links) or from other sites (external links) to help Bing discover new content and new pages.  
      
    Bing recommends linking all pages on a site to at least one other discoverable and crawlable page.
    
    * Crawlable links are <a> tags with an href attribute. The referring link should include either a text or an image alt attribute that is relevant to the page.
    * Limit the number of links on a page to a reasonable quantity, no more than a few thousand links per page.
    * Make a reasonable effort to ensure that any paid or advertisement links on your site use rel\="nofollow" or rel\="sponsored" or rel\="ugc" attribute to prevent the links from being followed by a crawler and from potentially impacting search rankings.
    * Bing rewards links that have grown organically; links added over time by content creators on other trusted, relevant websites that drive real users from their site to yours. Plan to build links both internally and externally, in an organic manner.
    
      
    Abusive tactics that aim to inflate the number and nature of inbound links, such as links buying participating in link schemes (link farms, link spamming and excessive link manipulation) can lead to your site being penalized and delisted from the Bing index.  
      
    
* **Limit the number of web pages:** Limit the number of pages on your website to a reasonable number. Avoid duplicative content within your site; help us dedupe duplicate content by:
    * Avoid outputting different URLs with the same content by using the [Canonical Tag](https://en.wikipedia.org/wiki/Canonical_link_element#Implementation).
    * Configuring your website and URL Parameters to improve crawl efficiency and help reduce multiple variations of the same URL pointing to the same content.
    * [Avoid mobile-specific URLs](https://blogs.bing.com/webmaster/2012/03/07/building-websites-optimized-for-all-platforms-desktop-mobile-etc/). Try using the same URL for desktop and mobile users.  
          
        
* **Use redirects as appropriate:** If you move content on your website to another location, use a [HTTP 301 permanent redirect](https://en.wikipedia.org/wiki/HTTP_301) for at least three months. If the move is temporary, i.e. less than one day, use a [302 temporary redirect](https://en.wikipedia.org/wiki/HTTP_302). Avoid using a [rel=canonical tag](https://en.wikipedia.org/wiki/Canonical_link_element) in place of a proper redirect when site content has moved from one location to another.  
      
    
* **Let Bing crawl more:** The Webmaster Tool's [crawl control feature](https://www.bing.com/webmasters/help/crawl-control-55a30303) allows you to manage how Bingbot crawls your content, including when, and at what pace. We encourage Webmasters to enable Bingbot to quickly and deeply crawl sites to ensure as much content as possible is discovered and indexed.  
      
    
* **JavaScript:**Bingcan process JavaScript, however, there are limitations to processing JavaScript at scale while minimizing the number of HTTP requests. Bing recommends [Dynamic Rendering](https://blogs.bing.com/webmaster/october-2018/bingbot-Series-JavaScript,-Dynamic-Rendering,-and-Cloaking-Oh-My) to switch between client-side rendered and pre-rendered content for specific user agents such as Bingbot, especially large web sites.  
      
    
* **Remove content** by returning a [404 "Not Found" HTTP code](https://en.wikipedia.org/wiki/HTTP_404). Expedite content removal by using the [Bing Content Removal and Page Removal tools](https://www.bing.com/webmasters/help/bing-content-removal-tool-cb6c294d). Content removal requests last for a maximum of 90 days, and you need to renew it, or content may reappear in the search results.  
      
    
* **robots.txt:** A robots.txt file informs search engine crawlers such as Bingbot, which pages and files the crawler can or cannot access on your site. Robots.txt is primarily used to instruct and manage crawler traffic, For example you can tell Bingbot not to crawl less helpful content such as search result pages or login pages.
    * Place the robots.txt in your root directory (the topmost directory) of your website. Do not place it in a subdirectory.
    * Blocking Bing from crawling a page will likely remove the page from the index. However, using Disallow does not guarantee that a page will not appear within the index or search results. If you would like to block a specific page from getting crawled or indexed, you should use the noindex robots meta tag instead of disallowing it in the robots.txt.
    * Review your robots.txt often to ensure it's up to date. Review URLs disallowed by robots.txt in Bing Webmaster Tools to ensure it remains accurate.
    * [Learn more by reading how to create a robots.txt text file](https://www.bing.com/webmasters/help/how-to-create-a-robots-txt-file-cb7c31ec)  
          
        
* **Save resources:** Use [HTTP Compression and Conditional Get](https://blogs.bing.com/webmaster/2008/02/12/announcing-crawler-improvements-for-live-search/) to reduce bandwidth used by the crawlers and your customers while improving page load speed.

Help Bing understand your pages
-------------------------------

Bing seeks rich, valuable, and engaging content created for users, not search engines. By creating clear, unique, high-quality, relevant, and easy-to-find content on your website, you will increase the probability of Bing indexing and showing your content in search results.

* **Content**: Websites that are thin on content, show ads or affiliate links primarily, or redirect visitors away to other sites quickly tend to drop in rankings on Bing. In some cases, it may not get indexed at all. Your content should be easy to navigate, rich and engaging to the website visitor, and provide the information they seek.  
      
    * **Create content for search users not search engines:** Develop rich content based on [keyword research](https://www.bing.com/webmasters/help/keyword-research-tool-628070b6) that shows what search users the information they are looking for.  
          
        
    * **Create enough content** to meet the visitor’s expectations entirely. There are no hard and fast rules on the number of words per page, but providing more relevant content is usually better..  
          
        
    * **Make it unique.** Do not reuse content from other sources. Content on your page must be unique in its final form. If you choose to host content from a third party, either use the canonical tag (rel\="canonical" to identify the original source or use the alternate tag (rel\="alternate") .  
          
        
    * **Images and Videos:** Use unique and original images and videos relevant to the topic of the page. Bing can extract information from images, captions, structured data, titles, and related text such as transcripts.
        
        * Do not embed important text or information within images or videos. Optical Character Recognition is less reliable than HTML text, and it is not accessible. [Alt text](https://en.wikipedia.org/wiki/Alt_attribute) improves accessibility for people and devices that cannot see images on pages. When choosing alt text, focus on creating information-rich content that uses keywords appropriately to give context to the image related to the content of the page.
        * Include descriptive titles, filenames, and text for images and videos.
        * The videos must be in a supported format and not blocked behind paywall protection or logins.
        * Subtitles and captions can be used for videos to help make your content available to a broader audience and provide search engines a textual representation of the content in video and audio files.
        * Choose high-quality photos and videos; they appeal more to users than blurry or out-of-focus images.
        
          
        
    * Optimize images and videos to improve page load times. Images are often the most significant contributor to page size and slower page loads.  
        
    * **SafeSearch** is a setting that specifies whether to show or hide explicit images, videos, and websites in search results. Bing uses machine learning to help categorize images. However the strongest signal is self-marked adult pages and content. We recommend you help Bing understand Adult-Only images and content by:
        
        * Using <meta name \="rating" content\="adult">
        * Grouping adult-only images in a common file folder: [http://www.example.com/adult/image.png](http://www.example.com/adult/image.png)
        
          
        
    * **Make your content discoverable.** Avoid housing content inside Flash or JavaScript – these block crawlers from finding content.  
          
        
    * **Make content** [**accessible and easy to navigate**](https://www.w3.org/WAI/standards-guidelines/wcag/) for all site visitors: Test the usability for devices such as screen readers. Listening to your content instead of looking at it can be eye-opening and help you check the accuracy and quality of the alt text for images and videos. Testing for usability can also help identify issues and problems with navigation, reading order, table markup and form elements within your website.
* **HTML Tags:** Ensure your HTML element and alt attributes are descriptive, specific, and accurate.
    
    * **<TITLE>** tags – Title of the page. Write descriptive and unique titles for each page of your website.  
          
        
    * **<META name="description">** – This is the summary and description of the webpage and may appear as the page description in the search results. Write relevant descriptions; you can use this space to expand on the <title> tag.  
          
        
    * **<META name="robots">** – You can provide crawling instructions to crawlers on indexing specific page content using these. Using [robots meta tag](https://blogs.bing.com/webmaster/april-2020/Announcing-new-options-for-webmasters-to-control-their-snippets-at-Bing), you can let Bing know about your snippet and content preview preferences.  
          
        
    * **<a href> tag** – specifies the URL linking to another page. If you want to link to another part on the same page use the #tag.  
          
        
    * **<img src> tag** –specifies an image file to be displayed  
          
        
    * **alt** attributes – use this attribute on <img> tags to describe the image. Use descriptive and information-rich context within alt attributes to provide context to the images.  
          
        
    * **<H1> tag** – helps users understand the content of a page more clearly when properly used.  
          
        
    * **<H1>-<H6> Header tags** – Define the structure of your page and helps Bing understand the content of each paragraph.  
          
        
    * **<P> tag** – delineates paragraphs.  
          
        
    * **<TABLE> tag** – Use <TABLE><TH> etc. for data tables. Do not use <TABLE> for layout.  
          
        
    * **Use HTML5 semantic elements** as they have an intrinsic meaning to browser, developer and search engine, especially use the following HTML5 Semantic Elements: <article>, <aside>, <details>, <figcaption>, <figure>, <footer>, <header>, <main>, <mark>, <nav>, <section>, <summary>, <time>.  
          
        
    * **Verify your HTML** via [URL Inspection tool](https://www.bing.com/webmasters/help/URL-Inspection-55a30305) or [Markup Validation Service](https://validator.w3.org/).  
          
        
* **Microsoft Edge**: Make sure your web page does not appear broken in the Microsoft Edge browser. When the document is loaded, searchable content must be visible, and there should be no pop-ups on page load.  
      
    
* **Secondary content (CSS, JavaScript)**: Allow crawlers in your robots.txt. to crawl all your CSS Style Sheet and JavaScript files. Limit usage of dynamic loading of resources – i.e., AJAX to limit the number of HTTP requests and limit the use of JavaScript on large websites.  
      
    
* **Use Semantic Markup** to convey information about the pages in either Schema.org, RDFa, or OpenGraph. Schema.org is preferred, in either JSON-LD or Microdata format. Note that Semantic Markup may enable Bing's rich features to be triggered. However it does not guarantee it. Verify Schema markup on your pages using [URL Inspection](https://www.bing.com/webmasters/help/URL-Inspection-55a30305).  
      
    

Abuse and Examples of Things to Avoid
=====================================

Search Engine Optimization is a legitimate practice that seeks to improve the technical and content aspects of a website, making the content easier to find, relevant, and more accessible to search engine crawlers. Most SEO practices render a website more appealing to Bing; however, performing SEO-related work does not guarantee rankings will improve or sites will receive an increase in traffic from Bing. Further, taken to extremes, some SEO practices can be abused, resulting in penalties by search engines.  

These Bing Webmaster Guidelines describe only some of the most widespread forms of inappropriate, manipulative, or misleading behaviors. Sites that engage in abuse, such as the practices outlined below, are considered low quality. As a result, these sites can incur ranking penalties, have site markup ignored, and may not be selected for indexing. Microsoft may take action against your site for any inappropriate or deceptive practices, even those not described here. If you feel action is taken against your site, you can contact our support team through the Bing Webmaster Tools. Additionally, users can report abuse of any of these practices using the feedback link in the footer of bing.com after performing a search that reproduces the issue. An example of this can be seen here.  

* **Cloaking:** Cloaking is the practice of showing one version of a webpage to a search crawler like Bingbot and another to regular visitors. Showing users different content than what the crawlers see may be seen as a spam tactic and could be detrimental to your website's rankings, leading your site to be de-listed from the Bing index. Websites should be extremely cautious about responding differently to crawlers as opposed to regular visitors and should not use cloaking as a principle.  
      
    
* **Link Schemes, Link Buying, Link Spamming:** While link schemes may succeed in increasing the number of links pointing to your website, they will fail to bring quality links to your site, netting your site no positive gains. Manipulating inbound links to artificially inflate the number of links pointing to a website can lead to your site being delisted from the Bing index.  
      
    
* **Social media schemes:** Social media schemes are similar to link farms in that they seek to artificially exploit a network effect to game Bing’s algorithm. The reality is that social media schemes are easy to see in action, and the website’s value is deprecated. One such example is auto follow. Auto follows encourage follower growth on social sites such as Twitter, and work by automatically following anyone who follows you. Over time this creates a scenario where the number of followers you have is more or less the same as the number of people following you. This does not indicate you have a strong influence. Following few people while having a high follower count would indicate a stronger influential voice.  
      
    
* **Duplicate content:** Duplicating content across multiple URLs can lead to Bing losing trust in some of those URLs over time. This issue should be managed by fixing the root cause of the problem. The rel=canonical element can also be used but should be considered a secondary solution to fixing the core problem. If excessive parameterization is causing the duplicate content issue, we encourage you to use the [URL Inspection tool](https://www.bing.com/webmasters/help/URL-Inspection-55a30305).  
      
    
* **Scraped Content:** Scraping or copying content from other more reputable websites may not add value to your users and may be considered copyright infringement. Instead of republishing content from other sites without adding any original content or value, you should create original content differentiating your website. Slightly modifying and republishing content from other websites also constitutes scraping.  
      
    
* **Keyword Stuffing OR loading pages with irrelevant keywords:** When creating content, make sure to create your content for real users and readers, not to entice search engines to rank your content better. Stuffing your content with specific keywords with the sole intent of artificially inflating the probability of ranking for particular search terms is in violation of our guidelines. It can lead to demotion or even delisting your website from our search results.  
      
    
* **Automatically generated content:** Machine-generated content is information that is generated by an automated computer process, application, or other mechanisms without any active intervention of a human. Content like this is considered malicious and usually contains garbage text only created to garnish a higher ranking. This type of content will result in penalties.  
      
    
* **Affiliate programs without adding sufficient value:** Websites that link products from other websites (such as Amazon, eBay, etc.) but pretend that they are an official retailer or in affiliation with those sites are called Thin Affiliation sites, and they add little to no incremental value (like extra reviews, search functionality, Editor’s Choice, etc.). and are only intended to take credit for other content. It can lead to demotion or even delisting your website from our search results.  
      
    
* **Malicious behavior:** When creating content and managing your site, make sure to not participate in phishing or installing viruses, trojans, or other badware by verifying your content, maintaining your Content Management System, keeping your operating system up to date, and limiting access to who can publish on your site. Malicious behavior can lead to demotion or even delisting your website from our search results.  
      
    
* **Misleading structured data markup:** Site with markups must be accurate and representative of the page that the tags are on. Sites must not have markup which is irrelevant to the page and is inaccurate or misleading.  
      
    
* **Prompt injection:** Do not add content on your webpages which attempts to perform prompt injection attacks on language models used by Bing. This can lead to demotion or even delisting of your website from our search results.

For information on how Bing ranks your content and on Bing's additional search experiences, please visit our [How Bing delivers search results](https://support.microsoft.com/topic/how-bing-delivers-search-results-d18fc815-ac37-4723-bc67-9229ce3eb6a3) page.
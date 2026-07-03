import asyncio
from playwright.async_api import async_playwright
import time

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("http://localhost:8501")
        await page.wait_for_selector('[data-testid="stFileUploaderDropzone"]')
        
        # Hover over the dropzone
        await page.hover('[data-testid="stFileUploaderDropzone"]')
        await asyncio.sleep(1)
        
        # Get the outer HTML of the entire stFileUploader
        html = await page.locator('[data-testid="stFileUploader"]').evaluate('el => el.outerHTML')
        with open("tests/uploader_dom.html", "w") as f:
            f.write(html)
            
        await browser.close()

asyncio.run(main())

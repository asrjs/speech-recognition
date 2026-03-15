from playwright.sync_api import sync_playwright
import time
import json

def test_model(model_name):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-webgpu'])
        page = browser.new_page()
        
        # Capture console
        page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))
        
        print(f"\n--- Testing {model_name} on WebGPU ---")
        page.goto("http://localhost:5174/")
        
        # Wait for page to be ready
        page.wait_for_selector("#backend")
        
        # Select webgpu
        page.select_option("#backend", "webgpu")
        
        # Set repoId to the local model alias
        page.fill("#repoId", model_name)
        
        # Click Load
        page.click("#loadBtn")
        
        # Wait for "Model loaded." in logs or check if singleBtn is enabled
        print("Loading...")
        page.wait_for_selector("#singleBtn:not([disabled])", timeout=60000)
        
        # Click Run Random Sample (or Run First Sample)
        page.click("#singleBtn")
        
        print("Running inference...")
        
        # Wait until metrics changes to contain "summary" or we see "complete" in logs
        for _ in range(60):
            time.sleep(0.5)
            metrics_text = page.inner_text("#metrics")
            if "summary" in metrics_text.lower():
                break
        
        time.sleep(1) # just slightly more to let DOM settle
        metrics_text = page.inner_text("#metrics")
        print("\nMetrics Output:")
        print(metrics_text)
        
        browser.close()

if __name__ == "__main__":
    test_model("local")
    test_model("local_fp16")
    test_model("local_int8")

import uvicorn
import sys

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🛒 SMART CART CUSTOMER CLUSTERING DASHBOARD")
    print("="*60)
    print("Initializing backend pipeline server...")
    print("Web dashboard will be served at: http://127.0.0.1:8000")
    print("="*60 + "\n")
    
    try:
        uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nStopping SmartCart server. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)

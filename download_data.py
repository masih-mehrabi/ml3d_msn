import requests
import time
import base64
def login_and_get_file():
    # Step 1: Send a POST request to login and get the cookie prefix and token
    login_url = "https://webdisk.ads.mwn.de/Default.aspx?ReturnUrl=%2f"
    
    encoded_login_data = "eyJ1c2VybmFtZSI6ImdlMjZodXYiLCAicGFzc3dvcmQiOiJDbXU2QCRzVk1uUl9EaVIifQ=="

    decoded_login_data = base64.b64decode(encoded_login_data).decode()
    login_data = eval(decoded_login_data)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    session = requests.Session()
    response = session.post(login_url, data=login_data, headers=headers)

    # Check if login was successful (you can add more checks based on the response)
    if response.status_code != 200 or "invalid credentials" in response.text.lower():
        print("Login failed. Please check your credentials.")
        return

    # Extract the prefix and token from the cookies
    cookies_dict = requests.utils.dict_from_cookiejar(response.cookies)
    prefix = cookies_dict.get(".ASPXAUTH", "")
    token = prefix.split("=", 1)[1] if "=" in prefix else ""
    print(prefix)
    print(token)

    # Step 2: Send a GET request to download the file using the saved cookie prefix and token
    download_url = f"https://webdisk.ads.mwn.de/Handlers/Download.ashx?file=Home%2Fmodelnet10.zip&action=download"
    headers = {
        "Cookie": f".ASPXAUTH={prefix}; token={token}"
    }

  # Retry and increase timeout
    max_retries = 3
    retry_delay = 5
    for retry in range(max_retries):
        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=60) as file_response:
                file_response.raise_for_status()  # Check for any HTTP error response
                if file_response.status_code == 200:
                    # Set the destination file path
                    destination = "/content/ml3d_msn/data/modelnet10.zip"
                    with open(destination, "wb") as f:
                        for chunk in file_response.iter_content(chunk_size=8192000):
                            f.write(chunk)
                    print(f"File downloaded and saved to {destination}.")
                    break  # Successful download, exit the loop
                else:
                    print("File download failed.")
        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            print(f"Error during download, retrying... (attempt {retry+1}/{max_retries})")
            time.sleep(retry_delay)
    else:
        print("Download failed after multiple retries.")

if __name__ == "__main__":
    login_and_get_file()
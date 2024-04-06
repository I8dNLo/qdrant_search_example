**Advertisement Image Dataset Search Tool**

Welcome to the Advertisement Image Dataset Search Tool! This tool allows you to search through a dataset of advertisement images using text prompts. The service leverages the VIT-CLIP model to embed both images and prompts into the same vector space, enabling efficient and accurate searching. The embedded images are stored in a Qdrant database, providing fast retrieval.

Getting Started
To get started with the Advertisement Image Dataset Search Tool, follow these steps:

Clone the repository to your local machine:

```
git clone https://github.com/your/repository.git
```
Navigate to the project directory:


```cd advertisement-image-search-tool```

Ensure you have Docker installed on your machine.

Run the following command to start the service:


```docker-compose up```

Usage
Once the service is up and running, you can access the search tool through your web browser at http://localhost:7860. The frontend interface allows you to enter text prompts to search through the advertisement image dataset.

Components
The Advertisement Image Dataset Search Tool consists of the following components:

**VIT-CLIP Model**: This model is used to embed both images and text prompts into the same vector space.

**Qdrant Database**: The embedded images are stored in a Qdrant database, providing efficient retrieval for search queries.

**Fill Database Script**: This script loads the initial vector database with the embedded image data.

**Frontend Service**: The frontend service provides a user-friendly interface for searching through the dataset using text prompts.

**Contributing**

Contributions to the Advertisement Image Dataset Search Tool are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. Be sure to follow the contribution guidelines outlined in the repository.

**License**

This project is licensed under the MIT License.

**Acknowledgements**

[OpenAI](https://openai.com/) for developing the VIT-CLIP model.

[Qdrant](https://qdrant.tech/) for providing the Qdrant database.

[Docker](https://www.docker.com/) for containerization technology.

[GitHub](https://github.com/) for hosting the repository.

**Support**

If you encounter any issues or have any questions, feel free to open an issue on the GitHub repository. We're here to help!

Enjoy using the Advertisement Image Dataset Search Tool! 🚀
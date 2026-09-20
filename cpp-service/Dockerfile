FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    g++ \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN g++ -std=c++17 -DCPPHTTPLIB_OPENSSL_SUPPORT newwithapi.cpp -o server -lssl -lcrypto -lpthread

EXPOSE 8080

CMD ["./server"]
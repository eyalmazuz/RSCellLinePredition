import torch
from torch import nn


class FeatureEmbedding(nn.Module):
    def __init__(self, num_categorical, num_numerical=0, k=50):
        super(FeatureEmbedding, self).__init__()

        self.cat_embeddings = nn.Embedding(num_categorical, k)
        if num_numerical != 0:
            weights = nn.init.xavier_uniform_(torch.zeros((k, num_numerical)))
            self.register_parameter('num_embedding', nn.Parameter(weights))

    def forward(self, cat_features, num_features=None):
        embds = self.cat_embeddings(cat_features)

        if num_features is not None:
            num_embds = torch.einsum('ik,jk->ikj', num_features, self.num_embedding)
            embds = torch.cat([embds, num_embds], dim=1)

        return embds


class FactorizationMachine(nn.Module):
    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, embds):
        sum_of_square = embds.pow(2).sum(dim=1)
        square_of_sum = embds.sum(dim=1).pow(2)

        interactions = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        return interactions


class LinearModel(nn.Module):
    def __init__(self, num_categorical, num_numerical=None):
        super(LinearModel, self).__init__()
        self.cat_linear = nn.Embedding(num_categorical, 1)
        if num_numerical != 0:
            weights = nn.init.xavier_uniform_(torch.zeros((1, num_numerical)))
            self.register_parameter('num_linear', nn.Parameter(weights))

        self.register_parameter('bias', torch.nn.Parameter(torch.zeros((1,))))

    def forward(self, cat_features, num_features=None):
        linear_weights = self.cat_linear(cat_features)
        if num_features is not None:
            num_linear_weights = torch.einsum('ik,jk->ikj', num_features, self.num_linear)
            linear_weights = torch.cat([linear_weights, num_linear_weights], dim=1)

        return linear_weights.sum(1) + self.bias


class FM(nn.Module):
    def __init__(self, categorical_dims, num_numerical=0, k=50):
        super(FM, self).__init__()

        self.embeddings = FeatureEmbedding(sum(categorical_dims) + 1, num_numerical, k)
        self.fm = FactorizationMachine()
        self.linear = LinearModel(sum(categorical_dims) + 1, num_numerical)

    def forward(self, cat_features, num_features=None):
        fm_term = self.fm(self.embeddings(cat_features, num_features))
        linear_term = self.linear(cat_features, num_features)

        return fm_term + linear_term


class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims, dropout):
        super(MLP, self).__init__()

        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_size, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            input_size = dim

        layers.append(nn.Linear(input_size, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DeepFM(nn.Module):
    def __init__(self, categorical_dims, num_numerical=0, k=50, hidden_dims=(512, 512),
                 dropout=0.1, use_numerical_embeddings=True):
        super(DeepFM, self).__init__()
        self.use_numerical_embeddings = use_numerical_embeddings

        self.embeddings = FeatureEmbedding(sum(categorical_dims) + 1, num_numerical, k)
        self.fm = FactorizationMachine()
        self.linear = LinearModel(sum(categorical_dims) + 1, num_numerical)

        self.mlp_input_size = (len(categorical_dims) + num_numerical) * k
        self.mlp = MLP(self.mlp_input_size, hidden_dims, dropout)

    def forward(self, cat_features, num_features=None):
        if self.use_numerical_embeddings:
            embeds = self.embeddings(cat_features, num_features)
        else:
            embeds = self.embeddings(cat_features)
        fm_term = self.fm(embeds)
        if self.use_numerical_embeddings:
            mlp_term = self.mlp(embeds.view(-1, self.mlp_input_size))
        else:
            mlp_input = torch.cat([embeds.view(-1, self.mlp_input_size), num_features], dim=-1)
            mlp_term = self.mlp(mlp_input)
        linear_term = self.linear(cat_features, num_features)

        return fm_term + linear_term + mlp_term

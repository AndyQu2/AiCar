from torch import optim


def evaluate(model, data_loader, criterion, device):
    model.eval()
    losses = []

    for source, target in data_loader:
        image = source.to(device)
        label = target.to(device)

        prediction = model(image)
        loss = criterion(prediction, label)
        losses.append(loss.item())

    return sum(losses) / len(losses)

def train(model, train_loader, criterion, optimizer, epochs, device, writer):
    best_model = None
    best_loss = float('inf')
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=7e-3)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for source, target in train_loader:
            image = source.to(device)
            label = target.to(device)

            result = model(image)
            loss = criterion(result, label)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        average_loss = sum(train_losses) / len(train_losses)

        if average_loss < best_loss:
            best_loss = average_loss
            best_model = model

        print(f"Epoch [{epoch}/{epochs}] Train Loss: {average_loss:.10f}")
        writer.add_scalar('train loss', average_loss, epoch)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

    return best_model
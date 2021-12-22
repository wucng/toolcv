- https://codechina.csdn.net/wc781708249/toolsmall

```py
"""advanced"""
def train_one_epoch_V2(model, optimizer, loss_func, data_loader, device, epoch,
                       print_freq=50, use_oneshot_loss=False, gamma=0.5, margin=2,
                       mixup_alpha=1.0, ricap_beta=0.3, seed=100):
    """
    # 设置一个随机数，来选择增强方式
    # 1.普通方式
    # 2.ricap
    # 3.mixup
    """
    np.random.seed(seed)
    model.train()
    num_datas = len(data_loader.dataset)
    for batch_idx, (images, targets) in enumerate(data_loader):
        state = np.random.choice(["general", "ricap", "mixup"], 1)[0]

        if state == "general":
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            losses = cls_loss(loss_func, outputs, targets, use_focal_loss, smooth_label)

            if use_oneshot_loss:
                # 结合one-shot-learning loss : 类内小 内间大
                outputs1, outputs2 = outputs[:-1], outputs[1:]
                label = (targets[:-1] == targets[1:]).float()  # 不同为0 相同为1
                losses_oneshot = oneshot_loss(outputs1, outputs2, label, margin)
                losses = losses + losses_oneshot * gamma

        elif state == "ricap":
            # ricap 数据随机裁剪组合增强
            I_x, I_y = images.size()[2:]

            w = int(np.round(I_x * np.random.beta(ricap_beta, ricap_beta)))
            h = int(np.round(I_y * np.random.beta(ricap_beta, ricap_beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(images.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = images[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = targets[idx].to(device)
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                 torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)
            patched_images = patched_images.to(device)
            output = model(patched_images)

            losses = sum([W_[k] * loss_func(output, c_[k]) for k in range(4)])

        else:  # mixup
            l = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(images.size(0))
            input_a, input_b = images, images[idx]
            target_a, target_b = targets, targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b

            target_a = target_a.to(device)
            target_b = target_b.to(device)
            mixed_input = mixed_input.to(device)
            output = model(mixed_input)
            losses = l * loss_func(output, target_a) + (1 - l) * loss_func(output, target_b)

        optimizer.zero_grad()
        losses.backward()
        # clip_grad_norm_(model.parameters(), 1.0 - 1e-10)  # 梯度裁剪
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), num_datas,
                       100. * batch_idx * len(images) / num_datas, losses.item()))

```
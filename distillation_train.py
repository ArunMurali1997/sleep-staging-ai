import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from train_models import CNN, EnhancedViT, evaluate, preprocess
from config.config import OUTPUT_DIR, DEVICE, EPOCHS, ALPHA, T, META_FILE


def train_distillation(train_loader, test_loader, class_weights):

    if not os.path.exists(META_FILE):
        print("Starting preprocessing")
        preprocess()

    print("\n=================================")
    print("Step 1: Load Teacher and Student")
    print("=================================")

    teacher = EnhancedViT().to(DEVICE)
    teacher.load_state_dict(torch.load(OUTPUT_DIR / "model_vit.pth"))
    teacher.eval()

    print("Teacher Model: ViT (Pretrained Loaded)")

    student = CNN().to(DEVICE)

    # Optional: start from pretrained CNN
    try:
        student.load_state_dict(torch.load(OUTPUT_DIR / "model_cnn.pth"))
        print("Student Model: CNN (Pretrained Loaded)")
    except:
        print("Student Model: CNN (Random Init)")

    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(student.parameters(), lr=3e-4)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    print("\nStep 2: Distillation Loss = KL Divergence")
    print("Step 3: Student vs Ground Truth")
    print("Step 4: Cross Entropy Loss")
    print("Step 5: Total Loss = α CE + (1-α) KL")
    print("Step 6: Backpropagate to CNN")

    for epoch in range(EPOCHS):

        student.train()

        total_loss = 0
        total_ce = 0
        total_kd = 0

        for x,y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            # Teacher prediction
            with torch.no_grad():
                teacher_logits = teacher(x)

            # Student prediction
            student_logits = student(x)

            # Cross entropy loss
            loss_ce = ce_loss(student_logits, y)

            # Distillation loss
            loss_kd = kl_loss(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            ) * (T*T)

            # Total loss
            loss = ALPHA * loss_ce + (1 - ALPHA) * loss_kd

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_kd += loss_kd.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"CE Loss: {total_ce/len(train_loader):.4f} | "
            f"KD Loss: {total_kd/len(train_loader):.4f} | "
            f"Total Loss: {total_loss/len(train_loader):.4f}"
        )

    print("\nStep 7: Final Sleep Stage Classification Evaluation")

    acc, f1, cm = evaluate(student, test_loader, "Distilled CNN")

    torch.save(student.state_dict(), OUTPUT_DIR / "model_cnn_distilled.pth")

    print("\nDistilled CNN saved to:")
    print(OUTPUT_DIR / "model_cnn_distilled.pth")

    print("\nFinal Results")
    print("Accuracy:", acc)
    print("Macro F1:", f1)

    return student
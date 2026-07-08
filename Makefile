# Makefile — UL-UNAS Fixed-Point Inference Engine
# Supports: PC (gcc) for testing, MIPS (mips-linux-gnu-gcc) for X2000

# ── Toolchain selection ──────────────────────────────────────────────────
# make TARGET=pc      → PC native build for golden testing
# make TARGET=x2000   → MIPS cross-compile for X2000

TARGET ?= pc

ifeq ($(TARGET), x2000)
    CC      = /home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
    CFLAGS  = -O2 -mips32r2 -msoft-float -Wall -Wextra -std=c99
    LDFLAGS = -lm -static
else
    CC      = gcc
    CFLAGS  = -O2 -g -Wall -Wextra -std=c99
    LDFLAGS = -lm
endif

# ── Directories ──────────────────────────────────────────────────────────
SRCDIR   = .
BUILDDIR = build

# ── Source files ─────────────────────────────────────────────────────────
SRCS  = ulunas_fp.c
SRCS += ulunas_lut.c
SRCS += ulunas_matlab_weights.c
SRCS += ulunas_modules.c
SRCS += ulunas_infer.c
SRCS += ulunas_stft.c

# Optional test programs
TEST_SRCS = test_matlab_golden.c
DIAG_SRCS = test_decoder_diag.c

# ── Objects ──────────────────────────────────────────────────────────────
OBJS      = $(patsubst %.c, $(BUILDDIR)/%.o, $(SRCS))
TEST_OBJS = $(patsubst %.c, $(BUILDDIR)/%.o, $(TEST_SRCS))

# ── Targets ──────────────────────────────────────────────────────────────

.PHONY: all clean test help

all: $(BUILDDIR)/libulunas.a

help:
	@echo "UL-UNAS Fixed-Point Inference Engine"
	@echo ""
	@echo "Targets:"
	@echo "  make [TARGET=pc]        Build static library (default: PC)"
	@echo "  make TARGET=x2000       Cross-compile for X2000 (MIPS32R2)"
	@echo "  make test               Build + run golden comparison test"
	@echo "  make clean              Remove build artifacts"
	@echo ""
	@echo "MATLAB prerequisites (run in MATLAB first):"
	@echo "  1. gen_lut_tables.m       → generates ulunas_lut.h/.c"
	@echo "  2. extract_weights.m      → generates ulunas_matlab_weights.h/.c"
	@echo "  3. export_all_layers.m    → generates golden_*.bin test data"

# ── Static library ───────────────────────────────────────────────────────
$(BUILDDIR)/libulunas.a: $(OBJS)
	@mkdir -p $(BUILDDIR)
	ar rcs $@ $^

# ── Compile ──────────────────────────────────────────────────────────────
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $< -o $@

# ── Test program ─────────────────────────────────────────────────────────
test: $(BUILDDIR)/test_matlab_golden
	@echo "Running golden comparison test..."
	$(BUILDDIR)/test_matlab_golden

$(BUILDDIR)/test_matlab_golden: $(OBJS) $(BUILDDIR)/test_matlab_golden.o
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(SRCDIR) $^ $(LDFLAGS) -o $@

diag: $(BUILDDIR)/test_decoder_diag
	@echo "Running Decoder diagnostic..."
	$(BUILDDIR)/test_decoder_diag

$(BUILDDIR)/test_decoder_diag: $(OBJS) $(BUILDDIR)/test_decoder_diag.o
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(SRCDIR) $^ $(LDFLAGS) -o $@

# ── Dependencies ─────────────────────────────────────────────────────────
$(BUILDDIR)/ulunas_fp.o:      ulunas_fp.h qr_config.h layer_dims.h ulunas_lut.h
$(BUILDDIR)/ulunas_lut.o:     ulunas_lut.h
$(BUILDDIR)/ulunas_matlab_weights.o: ulunas_matlab_weights.h
$(BUILDDIR)/ulunas_modules.o: ulunas_fp.h ulunas_matlab_weights.h qr_config.h layer_dims.h
$(BUILDDIR)/ulunas_infer.o:   ulunas_fp.h ulunas_matlab_weights.h
$(BUILDDIR)/test_matlab_golden.o: ulunas_fp.h
$(BUILDDIR)/test_decoder_diag.o: ulunas_fp.h

# ── Clean ────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILDDIR)

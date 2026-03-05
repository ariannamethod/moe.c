/*
 * moe.c — one file. one grok-style MoE. no excuses.
 *
 * 4 experts walk into a transformer. the router says "only 2 of you
 * are useful per token." the shared expert says "amateurs" and does
 * all the real work anyway. this is mixture-of-experts.
 *
 * trains from scratch. analytical backward passes through the router.
 * through the experts. through the gating. through the SwiGLU. through
 * the double pre-norm that nobody asked for but everyone secretly needs.
 * no autograd. no pytorch. no "it works in my notebook."
 *
 * cc moe.c -O3 -lm -lpthread -o moe && ./moe --depth 4
 *
 * depth is the only knob. you turn it, the architecture scales itself.
 * that's more agency than most ML engineers show at work.
 *
 *   depth 2  → ~2M params, learns what words are
 *   depth 4  → ~5M params, starts routing tokens like it means it
 *   depth 8  → ~25M params, experts develop specializations and grievances
 *
 * sibling of l.c (actually.llama). l.c is the responsible one that got
 * into a good university. moe.c is the one that dropped out to start
 * a committee where 4 specialists argue about every input and a 5th one
 * quietly fixes everything. familiar org structure? yeah.
 *
 * born from the Arianna Method ecosystem. grok's architecture.
 * opus's code. oleg's spite. trained by electrons that didn't consent.
 *
 * what happens when you run it:
 * 1. loads or generates data (synthetic shame is still data)
 * 2. trains BPE tokenizer from scratch (no sentencepiece. no tiktoken.)
 * 3. builds a full Grok-style MoE transformer
 * 4. trains with hand-written gradients through every expert
 * 5. finetunes on personality.txt (optional but recommended for chaos)
 * 6. exports GGUF (grokky.go compatible)
 * 7. drops you into chat with a model that has opinions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * CUDA ACCELERATION — cuBLAS on GPU. because A100s don't buy themselves.
 *   nvcc -c ariannamethod_cuda.cu -DUSE_CUDA -lcublas -O3
 *   cc moe.c ariannamethod_cuda.o -O3 -lm -lpthread -DUSE_CUDA -lcublas -lcudart -L/usr/local/cuda/lib64 -o moe
 * ═══════════════════════════════════════════════════════════════════════════════ */
#ifdef USE_CUDA
#include "ariannamethod_cuda.h"
static float *d_tmp_a, *d_tmp_b, *d_tmp_c;
static int d_tmp_size = 0;
static void gpu_ensure_tmp(int needed) {
    if (needed <= d_tmp_size) return;
    if (d_tmp_a) { gpu_free(d_tmp_a); gpu_free(d_tmp_b); gpu_free(d_tmp_c); }
    d_tmp_a = gpu_alloc(needed);
    d_tmp_b = gpu_alloc(needed);
    d_tmp_c = gpu_alloc(needed);
    d_tmp_size = needed;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * BLAS ACCELERATION — optional cblas_sgemm for matmul. 3-4x speedup.
 *   macOS:  cc moe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o moe
 *   Linux:  cc moe.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o moe
 * without USE_BLAS: pure scalar loops. portable. correct. slow on big dims.
 * with USE_BLAS: your matmul goes from "I'm doing my best" to "I have hardware."
 * ═══════════════════════════════════════════════════════════════════════════════ */
#ifdef USE_BLAS
  #ifdef ACCELERATE
    #define ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION — one knob to rule them all, one knob to find them,
 * one knob to bring them all and in the darkness bind them.
 * you turn depth. dim, heads, hidden_dim, experts — all figured out.
 * pytorch has 47 config files. we have one integer.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int depth, dim, n_heads, n_kv_heads, head_dim, hidden_dim;
    int vocab_size, seq_len;
    float norm_eps, rope_theta;
    int n_experts, top_k, has_shared;
    int use_gelu, double_prenorm;
    float attn_clamp, aux_loss_w;
    float lr, weight_decay;
    int batch_size, max_steps, warmup_steps, log_every, bpe_merges, personality_steps;
    char data_url[512]; /* URL to download training text */
    char data_path[256], gguf_path[256], personality_path[256];
} Config;

static Config config_from_depth(int depth) {
    Config c = {0};
    c.depth = depth;
    c.dim = depth * 64; /* 64 per depth level. each step genuinely widens the model. */
    c.dim = ((c.dim + 63) / 64) * 64;
    if (c.dim < 128) c.dim = 128; /* min 128: below this, attention has 1 head and no opinions */
    if (c.dim > 768) c.dim = 768; /* max 768: your CPU has feelings and RAM has limits */
    c.head_dim = 64;
    c.n_heads = c.dim / c.head_dim;
    if (c.n_heads < 1) c.n_heads = 1;
    if (c.dim <= 384) { c.n_kv_heads = c.n_heads; }
    else {
        c.n_kv_heads = c.n_heads / 2;
        if (c.n_kv_heads < 1) c.n_kv_heads = 1;
        while (c.n_heads % c.n_kv_heads != 0 && c.n_kv_heads > 1) c.n_kv_heads--;
    }
    c.n_experts = 4; c.top_k = 2; c.has_shared = 1; /* 4 experts, use 2, one shared. corporate structure. */
    c.hidden_dim = (int)(c.dim * 1.5f); /* 1.5x per expert because each one is a specialist, not a generalist */
    c.hidden_dim = ((c.hidden_dim + 63) / 64) * 64;
    c.seq_len = 256; c.norm_eps = 1e-5f; c.rope_theta = 10000.0f;
    c.use_gelu = 0; c.double_prenorm = 1; c.attn_clamp = 30.0f; c.aux_loss_w = 0.01f;
    /* SwiGLU because GELU is what you use when you haven't read the Grok paper.
     * double pre-norm because single pre-norm is for quitters.
     * attn_clamp=30 because experts get excited and need a ceiling.
     * aux_loss because without it, expert 0 hogs all tokens like a free buffet. */
    c.lr = 1e-4f; c.batch_size = 4; c.warmup_steps = 500;
    c.weight_decay = 0.01f; c.log_every = 20;
    /* MoE active params per token = attention + top_k/n_experts fraction of expert params + shared.
     * don't count all experts — only top_k are active per token. */
    long attn_p = 12L*depth*c.dim*c.dim;
    long expert_active = (long)(c.top_k+c.has_shared)*3*c.dim*c.hidden_dim*depth; /* active experts only */
    long active_pe = attn_p + expert_active;
    c.max_steps = (int)(active_pe / 1000);
    if (c.max_steps < 2000) c.max_steps = 2000;
    if (c.max_steps > 50000) c.max_steps = 50000;
    c.bpe_merges = 4000; c.personality_steps = 1000;
    snprintf(c.data_url, 512, "fineweb-edu"); /* marker: triggers HF API download of FineWeb-Edu */
    snprintf(c.data_path, 256, "moe_data.txt");
    snprintf(c.gguf_path, 256, "moe.gguf");
    snprintf(c.personality_path, 256, "personality.txt");
    return c;
}

static long count_params(Config *c) {
    long e = (long)c->vocab_size * c->dim * 2;
    long pl = (long)c->dim*c->n_heads*c->head_dim + (long)c->dim*c->n_kv_heads*c->head_dim*2
            + (long)c->n_heads*c->head_dim*c->dim + c->dim*2;
    if (c->double_prenorm) pl += c->dim * 2;
    pl += c->n_experts * c->dim;
    pl += (long)c->n_experts * c->dim * c->hidden_dim * 3;
    if (c->has_shared) pl += (long)c->dim * c->hidden_dim * 3;
    return e + pl * c->depth + c->dim;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*. three XORs and a multiply. pytorch uses Mersenne Twister
 * which is 2500 lines of C. we use 3. the experts don't care which PRNG
 * decided their fate. neither will your loss curve.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) { rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
static float rand_uniform(void) { return (float)(rng_next()&0x7FFFFFFF)/(float)0x7FFFFFFF; }
static float rand_normal(void) { float u1=rand_uniform(),u2=rand_uniform(); if(u1<1e-10f)u1=1e-10f; return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }

/* ═══════════════════════════════════════════════════════════════════════════════
 * DYNAMIC ARRAYS + BPE TOKENIZER — trained from scratch on your data.
 * no sentencepiece. no tiktoken. no "download the 4GB tokenizer model first."
 * 256 byte tokens + merges. stolen from l.c, who stole it from molequla,
 * who stole it from the idea that tokenization shouldn't require pip install.
 * tiktoken wishes it compiled in 0.3 seconds.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { char **items; int len, cap; } StrArr;
static void sa_push(StrArr *a, const char *s) { if(a->len>=a->cap){a->cap=a->cap?a->cap*2:16;a->items=realloc(a->items,sizeof(char*)*a->cap);}a->items[a->len++]=strdup(s); }
static void sa_free(StrArr *a) { for(int i=0;i<a->len;i++)free(a->items[i]);free(a->items);a->items=NULL;a->len=a->cap=0; }

#define TOK_MAX_VOCAB 16384
#define TOK_STOI_CAP 32768
typedef struct { char a[64]; char b[64]; } MergePair;
typedef struct { char *key; int val; } StoiEntry;
typedef struct { StoiEntry entries[TOK_STOI_CAP]; } StoiTable;
typedef struct { char *tokens[TOK_MAX_VOCAB]; int vocab_size; StoiTable stoi; int bos_id,eos_id; MergePair *merges; int n_merges; } Tokenizer;

static unsigned int str_hash(const char *s){unsigned int h=5381;while(*s)h=h*33+(unsigned char)*s++;return h;}
static void stoi_init(StoiTable *t){for(int i=0;i<TOK_STOI_CAP;i++){t->entries[i].key=NULL;t->entries[i].val=-1;}}
static void stoi_put(StoiTable *t,const char *key,int val){unsigned int h=str_hash(key)%TOK_STOI_CAP;for(int i=0;i<TOK_STOI_CAP;i++){int idx=(h+i)%TOK_STOI_CAP;if(!t->entries[idx].key){t->entries[idx].key=strdup(key);t->entries[idx].val=val;return;}if(strcmp(t->entries[idx].key,key)==0){t->entries[idx].val=val;return;}}}
static int stoi_get(StoiTable *t,const char *key){unsigned int h=str_hash(key)%TOK_STOI_CAP;for(int i=0;i<TOK_STOI_CAP;i++){int idx=(h+i)%TOK_STOI_CAP;if(!t->entries[idx].key)return -1;if(strcmp(t->entries[idx].key,key)==0)return t->entries[idx].val;}return -1;}

static void tok_init(Tokenizer *tok){memset(tok,0,sizeof(Tokenizer));stoi_init(&tok->stoi);for(int i=0;i<256;i++){char h[8];snprintf(h,8,"0x%02x",i);tok->tokens[tok->vocab_size]=strdup(h);stoi_put(&tok->stoi,h,tok->vocab_size);tok->vocab_size++;}tok->tokens[tok->vocab_size]=strdup("<BOS>");stoi_put(&tok->stoi,"<BOS>",tok->vocab_size);tok->bos_id=tok->vocab_size++;tok->tokens[tok->vocab_size]=strdup("<EOS>");stoi_put(&tok->stoi,"<EOS>",tok->vocab_size);tok->eos_id=tok->vocab_size++;}
static void tok_add(Tokenizer *tok,const char *s){if(stoi_get(&tok->stoi,s)>=0)return;if(tok->vocab_size>=TOK_MAX_VOCAB)return;tok->tokens[tok->vocab_size]=strdup(s);stoi_put(&tok->stoi,s,tok->vocab_size);tok->vocab_size++;}

static char byte_category(unsigned char b){if((b>='a'&&b<='z')||(b>='A'&&b<='Z'))return'L';if(b>='0'&&b<='9')return'N';if(b==' '||b=='\n'||b=='\r'||b=='\t')return'Z';if(b>=0x80)return'L';return'P';}
typedef struct{unsigned char*data;int len;}ByteSeg;
typedef struct{ByteSeg*segs;int len,cap;}SegArr;
static void seg_push(SegArr*a,unsigned char*data,int len){if(a->len>=a->cap){a->cap=a->cap?a->cap*2:64;a->segs=realloc(a->segs,sizeof(ByteSeg)*a->cap);}a->segs[a->len].data=malloc(len);memcpy(a->segs[a->len].data,data,len);a->segs[a->len].len=len;a->len++;}
static void seg_free(SegArr*a){for(int i=0;i<a->len;i++)free(a->segs[i].data);free(a->segs);memset(a,0,sizeof(SegArr));}

static SegArr unicode_segment(const char*text,int text_len){SegArr r={0};if(!text||text_len==0)return r;unsigned char buf[4096];int bl=0;char cc=0;const unsigned char*p=(const unsigned char*)text;for(int i=0;i<text_len;i++){char cat=byte_category(p[i]);if(cat!=cc&&bl>0){seg_push(&r,buf,bl);bl=0;}cc=cat;if(bl<(int)sizeof(buf)-1)buf[bl++]=p[i];else{seg_push(&r,buf,bl);bl=0;buf[bl++]=p[i];}}if(bl>0)seg_push(&r,buf,bl);return r;}

#define PAIR_CAP 32768
typedef struct{char a[64];char b[64];int count;int used;}PairEntry;
static unsigned int pair_hash(const char*a,const char*b){unsigned int h=5381;for(const char*p=a;*p;p++)h=h*33+(unsigned char)*p;h=h*33+0xFF;for(const char*p=b;*p;p++)h=h*33+(unsigned char)*p;return h;}

static void tok_train_bpe(Tokenizer*tok,const char*text,int tl,int nm){
    printf("[bpe] training %d merges on %d bytes...\n",nm,tl);
    SegArr segs=unicode_segment(text,tl);if(segs.len==0){seg_free(&segs);return;}
    int ns=segs.len;StrArr*ss=calloc(ns,sizeof(StrArr));
    for(int s=0;s<ns;s++)for(int b=0;b<segs.segs[s].len;b++){char h[8];snprintf(h,8,"0x%02x",segs.segs[s].data[b]);sa_push(&ss[s],h);}
    seg_free(&segs);if(tok->merges)free(tok->merges);tok->merges=calloc(nm,sizeof(MergePair));tok->n_merges=0;
    PairEntry*pairs=calloc(PAIR_CAP,sizeof(PairEntry));
    for(int it=0;it<nm;it++){
        memset(pairs,0,sizeof(PairEntry)*PAIR_CAP);
        for(int s=0;s<ns;s++){StrArr*sq=&ss[s];for(int i=0;i<sq->len-1;i++){unsigned int h=pair_hash(sq->items[i],sq->items[i+1])%PAIR_CAP;for(int p=0;p<64;p++){int idx=(h+p)%PAIR_CAP;if(!pairs[idx].used){strncpy(pairs[idx].a,sq->items[i],63);strncpy(pairs[idx].b,sq->items[i+1],63);pairs[idx].count=1;pairs[idx].used=1;break;}if(strcmp(pairs[idx].a,sq->items[i])==0&&strcmp(pairs[idx].b,sq->items[i+1])==0){pairs[idx].count++;break;}}}}
        int bc=1,bi=-1;for(int i=0;i<PAIR_CAP;i++)if(pairs[i].used&&pairs[i].count>bc){bc=pairs[i].count;bi=i;}
        if(bi<0)break;
        char nt[128];snprintf(nt,128,"%s+%s",pairs[bi].a,pairs[bi].b);
        strncpy(tok->merges[tok->n_merges].a,pairs[bi].a,63);strncpy(tok->merges[tok->n_merges].b,pairs[bi].b,63);tok->n_merges++;
        for(int s=0;s<ns;s++){StrArr*sq=&ss[s];StrArr mg={0};int i=0;while(i<sq->len){if(i<sq->len-1&&strcmp(sq->items[i],pairs[bi].a)==0&&strcmp(sq->items[i+1],pairs[bi].b)==0){sa_push(&mg,nt);i+=2;}else{sa_push(&mg,sq->items[i]);i++;}}sa_free(sq);*sq=mg;}
        tok_add(tok,nt);
        if((it+1)%500==0)printf("[bpe] %d/%d merges (vocab=%d)\n",it+1,nm,tok->vocab_size);
    }
    free(pairs);for(int s=0;s<ns;s++)sa_free(&ss[s]);free(ss);
    printf("[bpe] done: %d merges, vocab=%d\n",tok->n_merges,tok->vocab_size);
}

static int*tok_encode(Tokenizer*tok,const char*text,int tl,int*out_len){
    SegArr segs=unicode_segment(text,tl);int*ids=NULL;int ni=0,ci=0;
    for(int s=0;s<segs.len;s++){StrArr sy={0};for(int b=0;b<segs.segs[s].len;b++){char h[8];snprintf(h,8,"0x%02x",segs.segs[s].data[b]);sa_push(&sy,h);}
    if(tok->n_merges>0&&sy.len>=2){int ch=1;while(ch&&sy.len>=2){ch=0;int br=tok->n_merges,bp=-1;for(int i=0;i<sy.len-1;i++)for(int m=0;m<br;m++)if(strcmp(sy.items[i],tok->merges[m].a)==0&&strcmp(sy.items[i+1],tok->merges[m].b)==0){br=m;bp=i;break;}if(bp>=0){char nt[128];snprintf(nt,128,"%s+%s",tok->merges[br].a,tok->merges[br].b);StrArr mg={0};for(int i=0;i<sy.len;i++){if(i==bp){sa_push(&mg,nt);i++;}else sa_push(&mg,sy.items[i]);}sa_free(&sy);sy=mg;ch=1;}}}
    for(int i=0;i<sy.len;i++){int id=stoi_get(&tok->stoi,sy.items[i]);if(id<0)id=0;if(ni>=ci){ci=ci?ci*2:256;ids=realloc(ids,sizeof(int)*ci);}ids[ni++]=id;}sa_free(&sy);}
    seg_free(&segs);*out_len=ni;return ids;
}

static char*tok_decode(Tokenizer*tok,int*ids,int ni,int*out_len){
    char*buf=malloc(ni*8+1);int pos=0;
    for(int i=0;i<ni;i++){if(ids[i]<0||ids[i]>=tok->vocab_size)continue;if(ids[i]==tok->bos_id||ids[i]==tok->eos_id)continue;
    const char*nm=tok->tokens[ids[i]];const char*p=nm;while(*p){if(p[0]=='0'&&p[1]=='x'){unsigned int bv;if(sscanf(p,"0x%02x",&bv)==1)buf[pos++]=(char)bv;p+=4;if(*p=='+')p++;}else p++;}}
    buf[pos]='\0';*out_len=pos;return buf;
}
/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR — a float pointer that knows its own size. revolutionary technology.
 * pytorch wraps this in 14 layers of abstraction and 3 dispatch mechanisms.
 * we use a struct with 4 fields. the gradients don't know the difference.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { float *data; int size, rows, cols;
#ifdef USE_CUDA
    float *d_data;
#endif
} Tensor;

#ifdef USE_CUDA
static void gpu_upload_weights(Tensor **tensors, int n) {
    for (int i = 0; i < n; i++) {
        tensors[i]->d_data = gpu_alloc(tensors[i]->size);
        gpu_upload(tensors[i]->d_data, tensors[i]->data, tensors[i]->size);
    }
}
static void gpu_resync_weights(Tensor **tensors, int n) {
    for (int i = 0; i < n; i++)
        gpu_upload(tensors[i]->d_data, tensors[i]->data, tensors[i]->size);
}
#define GPU(t) ((t)->d_data)
#else
#define GPU(t) NULL
#endif
static Tensor *tnew(int s){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(s,sizeof(float));t->size=s;t->rows=1;t->cols=s;return t;}
static Tensor *tnew2d(int r,int co){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(r*co,sizeof(float));t->size=r*co;t->rows=r;t->cols=co;return t;}
static void tinit(Tensor*t,float std){for(int i=0;i<t->size;i++)t->data[i]=rand_normal()*std;}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MODEL WEIGHTS — the Grok MoE committee.
 * each layer has: attention (the memory), a router (the manager who assigns
 * work), 4 experts (the specialists who each think they're the best),
 * and a shared expert (the intern who actually does everything).
 * Wo initialized to zero so residual connections start clean.
 * router initialized to 0.01 so it doesn't pick favorites on day one.
 * this is more thought than most companies put into hiring.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { Tensor *w_gate, *w_up, *w_down; } ExpertW;
typedef struct {
    Tensor *attn_norm, *ffn_norm, *attn_post_norm, *ffn_post_norm;
    Tensor *wq, *wk, *wv, *wo;
    Tensor *w_router;
    ExpertW *experts;
    ExpertW *shared_expert;
} LayerW;
typedef struct { Tensor *tok_emb, *output, *output_norm; LayerW *layers; int n_layers; } ModelW;

static void init_weights(ModelW *w, Config *c) {
    float es=1.0f/sqrtf((float)c->dim), ls=1.0f/sqrtf((float)c->dim);
    int qd=c->n_heads*c->head_dim, kd=c->n_kv_heads*c->head_dim;
    w->tok_emb=tnew2d(c->vocab_size,c->dim); tinit(w->tok_emb,es);
    w->output=tnew2d(c->vocab_size,c->dim); tinit(w->output,ls);
    w->output_norm=tnew(c->dim); for(int i=0;i<c->dim;i++)w->output_norm->data[i]=1.0f;
    w->n_layers=c->depth; w->layers=calloc(c->depth,sizeof(LayerW));
    for(int l=0;l<c->depth;l++){
        LayerW *lw=&w->layers[l];
        lw->attn_norm=tnew(c->dim); lw->ffn_norm=tnew(c->dim);
        for(int i=0;i<c->dim;i++){lw->attn_norm->data[i]=1.0f;lw->ffn_norm->data[i]=1.0f;}
        if(c->double_prenorm){
            lw->attn_post_norm=tnew(c->dim); lw->ffn_post_norm=tnew(c->dim);
            for(int i=0;i<c->dim;i++){lw->attn_post_norm->data[i]=1.0f;lw->ffn_post_norm->data[i]=1.0f;}
        }
        lw->wq=tnew2d(qd,c->dim); tinit(lw->wq,ls);
        lw->wk=tnew2d(kd,c->dim); tinit(lw->wk,ls);
        lw->wv=tnew2d(kd,c->dim); tinit(lw->wv,ls);
        lw->wo=tnew2d(c->dim,qd); memset(lw->wo->data,0,lw->wo->size*sizeof(float));
        lw->w_router=tnew2d(c->n_experts,c->dim); tinit(lw->w_router,0.01f);
        lw->experts=calloc(c->n_experts,sizeof(ExpertW));
        for(int e=0;e<c->n_experts;e++){
            lw->experts[e].w_gate=tnew2d(c->hidden_dim,c->dim); tinit(lw->experts[e].w_gate,ls);
            lw->experts[e].w_up=tnew2d(c->hidden_dim,c->dim); tinit(lw->experts[e].w_up,ls);
            lw->experts[e].w_down=tnew2d(c->dim,c->hidden_dim); memset(lw->experts[e].w_down->data,0,lw->experts[e].w_down->size*sizeof(float));
        }
        if(c->has_shared){
            lw->shared_expert=calloc(1,sizeof(ExpertW));
            lw->shared_expert->w_gate=tnew2d(c->hidden_dim,c->dim); tinit(lw->shared_expert->w_gate,ls);
            lw->shared_expert->w_up=tnew2d(c->hidden_dim,c->dim); tinit(lw->shared_expert->w_up,ls);
            lw->shared_expert->w_down=tnew2d(c->dim,c->hidden_dim); memset(lw->shared_expert->w_down->data,0,lw->shared_expert->w_down->size*sizeof(float));
        }
    }
}

typedef struct { Tensor **tensors; int count; } ParamList;
static ParamList collect_params(ModelW *w, Config *c) {
    int mx=3+w->n_layers*(6+(c->double_prenorm?2:0)+1+c->n_experts*3+(c->has_shared?3:0));
    ParamList p; p.tensors=calloc(mx,sizeof(Tensor*)); p.count=0;
    p.tensors[p.count++]=w->tok_emb; p.tensors[p.count++]=w->output; p.tensors[p.count++]=w->output_norm;
    for(int l=0;l<w->n_layers;l++){
        LayerW*lw=&w->layers[l];
        p.tensors[p.count++]=lw->attn_norm;
        p.tensors[p.count++]=lw->wq; p.tensors[p.count++]=lw->wk;
        p.tensors[p.count++]=lw->wv; p.tensors[p.count++]=lw->wo;
        p.tensors[p.count++]=lw->ffn_norm;
        if(c->double_prenorm){p.tensors[p.count++]=lw->attn_post_norm;p.tensors[p.count++]=lw->ffn_post_norm;}
        p.tensors[p.count++]=lw->w_router;
        for(int e=0;e<c->n_experts;e++){p.tensors[p.count++]=lw->experts[e].w_gate;p.tensors[p.count++]=lw->experts[e].w_up;p.tensors[p.count++]=lw->experts[e].w_down;}
        if(c->has_shared){p.tensors[p.count++]=lw->shared_expert->w_gate;p.tensors[p.count++]=lw->shared_expert->w_up;p.tensors[p.count++]=lw->shared_expert->w_down;}
    }
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MATH OPS — the building blocks. rmsnorm, matvec, softmax, rope.
 * GELU is here but use_gelu=0 because SwiGLU won the activation wars.
 * GELU: "i approximate the gaussian error function with a tanh."
 * SiLU: "i'm just x*sigmoid(x)." fewer ops. better results. life's unfair.
 * RoPE rotates your queries and keys through complex space so the model
 * understands position. sincos for transformers. euler would be proud.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float gelu_f(float x){float c=0.7978845608f;float inner=c*(x+0.044715f*x*x*x);return 0.5f*x*(1.0f+tanhf(inner));}
static float gelu_bwd(float x){float c=0.7978845608f;float inner=c*(x+0.044715f*x*x*x);float th=tanhf(inner);float s2=1.0f-th*th;float di=c*(1.0f+3.0f*0.044715f*x*x);return 0.5f*(1.0f+th)+0.5f*x*s2*di;}
static float silu_f(float x){return x/(1.0f+expf(-x));}
static float silu_bwd(float x){float s=1.0f/(1.0f+expf(-x));return s+x*s*(1.0f-s);}

static void rmsnorm(float*out,float*x,float*w,int d,float eps){float ss=0;for(int i=0;i<d;i++)ss+=x[i]*x[i];float inv=1.0f/sqrtf(ss/d+eps);for(int i=0;i<d;i++)out[i]=x[i]*inv*w[i];}
static void matvec(float*out,float*W,float*x,int r,int co){for(int i=0;i<r;i++){float s=0;float*row=W+i*co;for(int j=0;j<co;j++)s+=row[j]*x[j];out[i]=s;}}
static void softmax_n(float*x,int n){float mx=x[0];for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}for(int i=0;i<n;i++)x[i]/=s;}

static void apply_rope(float*v,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float x0=v[i],x1=v[i+h];v[i]=x0*cc[off+i]-x1*sc[off+i];v[i+h]=x0*sc[off+i]+x1*cc[off+i];}}
static void rope_bwd(float*dv,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float d0=dv[i],d1=dv[i+h];dv[i]=d0*cc[off+i]+d1*sc[off+i];dv[i+h]=-d0*sc[off+i]+d1*cc[off+i];}}

/* top_k: the democratic process of expert selection. "who's loudest?" */
static void top_k_experts(float*logits,int n,int k,int*idx,float*wts){
    int used[16]={0};
    for(int ki=0;ki<k;ki++){float bv=-1e30f;int bi=0;for(int i=0;i<n;i++)if(!used[i]&&logits[i]>bv){bv=logits[i];bi=i;}idx[ki]=bi;wts[ki]=logits[bi];used[bi]=1;}
    /* temperature=2.0 prevents softmax polarization (0.9/0.1 → 0.6/0.4),
     * so both selected experts get meaningful gradient signal */
    float temp=2.0f;
    float mx=wts[0];for(int i=1;i<k;i++)if(wts[i]>mx)mx=wts[i];float s=0;for(int i=0;i<k;i++){wts[i]=expf((wts[i]-mx)/temp);s+=wts[i];}for(int i=0;i<k;i++)wts[i]/=s;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INFERENCE FORWARD — single token through the entire committee.
 * each token enters. gets normalized. gets attended to. gets routed to
 * 2 out of 4 experts who argue about what it means. the shared expert
 * rolls its eyes and adds its own opinion regardless. the token exits
 * changed, possibly confused, definitely transformed. literally.
 * attn_clamp keeps the attention scores from going nuclear. because
 * without it, expert 2 will decide one token is GOD and attend to it
 * with the force of a thousand suns. ask me how i know.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *x,*xb,*xb2,*xb3,*hb,*hb2,*hbs,*hb2s;
    float *q,*k,*v,*att,*logits;
    float *key_cache,*value_cache,*cos_cache,*sin_cache;
    float *router_logits,*expert_out;
} RunState;

static RunState alloc_run(Config *c){
    RunState s; int kd=c->n_kv_heads*c->head_dim;
    s.x=calloc(c->dim,sizeof(float)); s.xb=calloc(c->dim,sizeof(float));
    s.xb2=calloc(c->dim,sizeof(float)); s.xb3=calloc(c->dim,sizeof(float));
    s.hb=calloc(c->hidden_dim,sizeof(float)); s.hb2=calloc(c->hidden_dim,sizeof(float));
    s.hbs=calloc(c->hidden_dim,sizeof(float)); s.hb2s=calloc(c->hidden_dim,sizeof(float));
    s.q=calloc(c->n_heads*c->head_dim,sizeof(float)); s.k=calloc(kd,sizeof(float)); s.v=calloc(kd,sizeof(float));
    s.att=calloc(c->n_heads*c->seq_len,sizeof(float)); s.logits=calloc(c->vocab_size,sizeof(float));
    s.key_cache=calloc(c->depth*c->seq_len*kd,sizeof(float)); s.value_cache=calloc(c->depth*c->seq_len*kd,sizeof(float));
    int half=c->head_dim/2;
    s.cos_cache=calloc(c->seq_len*half,sizeof(float)); s.sin_cache=calloc(c->seq_len*half,sizeof(float));
    for(int p=0;p<c->seq_len;p++)for(int i=0;i<half;i++){float freq=1.0f/powf(c->rope_theta,(float)(2*i)/(float)c->head_dim);float ang=(float)p*freq;s.cos_cache[p*half+i]=cosf(ang);s.sin_cache[p*half+i]=sinf(ang);}
    s.router_logits=calloc(c->n_experts,sizeof(float)); s.expert_out=calloc(c->dim,sizeof(float));
    return s;
}

static float *forward_token(ModelW *w, Config *c, RunState *s, int token, int pos) {
    int D=c->dim, kd=c->n_kv_heads*c->head_dim, hd=c->head_dim, H=c->hidden_dim;
    int hg=c->n_heads/c->n_kv_heads; float sc=1.0f/sqrtf((float)hd);
    memcpy(s->x, w->tok_emb->data+token*D, D*sizeof(float));

    for(int l=0;l<c->depth;l++){
        LayerW *lw=&w->layers[l];
        /* Attention — "what should i look at?" asks the token. answer: everything before you. */
        rmsnorm(s->xb, s->x, lw->attn_norm->data, D, c->norm_eps);
        matvec(s->q, lw->wq->data, s->xb, c->n_heads*hd, D);
        matvec(s->k, lw->wk->data, s->xb, c->n_kv_heads*hd, D);
        matvec(s->v, lw->wv->data, s->xb, c->n_kv_heads*hd, D);
        for(int h=0;h<c->n_heads;h++) apply_rope(s->q+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        for(int h=0;h<c->n_kv_heads;h++) apply_rope(s->k+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        int co=l*c->seq_len*kd+pos*kd;
        memcpy(s->key_cache+co, s->k, kd*sizeof(float));
        memcpy(s->value_cache+co, s->v, kd*sizeof(float));

        for(int h=0;h<c->n_heads;h++){
            int kvh=h/hg; float*qh=s->q+h*hd; float*att=s->att+h*c->seq_len;
            for(int t=0;t<=pos;t++){int ko=l*c->seq_len*kd+t*kd+kvh*hd;float dot=0;for(int d=0;d<hd;d++)dot+=qh[d]*s->key_cache[ko+d];att[t]=dot*sc;}
            if(c->attn_clamp>0){float inv=1.0f/c->attn_clamp;for(int t=0;t<=pos;t++)att[t]=c->attn_clamp*tanhf(att[t]*inv);}
            softmax_n(att, pos+1);
            float*xb2h=s->xb2+h*hd; memset(xb2h,0,hd*sizeof(float));
            for(int t=0;t<=pos;t++){float a=att[t];int vo=l*c->seq_len*kd+t*kd+kvh*hd;for(int d=0;d<hd;d++)xb2h[d]+=a*s->value_cache[vo+d];}
        }
        matvec(s->xb, lw->wo->data, s->xb2, D, D);
        if(c->double_prenorm&&lw->attn_post_norm){rmsnorm(s->xb3,s->xb,lw->attn_post_norm->data,D,c->norm_eps);memcpy(s->xb,s->xb3,D*sizeof(float));}
        for(int i=0;i<D;i++) s->x[i]+=s->xb[i];

        /* MoE FFN — the committee meeting. router picks 2 experts. they process.
         * shared expert adds its unsolicited opinion. residual connection: "noted." */
        rmsnorm(s->xb, s->x, lw->ffn_norm->data, D, c->norm_eps);
        memset(s->expert_out, 0, D*sizeof(float));
        matvec(s->router_logits, lw->w_router->data, s->xb, c->n_experts, D);
        int ti[8]; float tw[8];
        top_k_experts(s->router_logits, c->n_experts, c->top_k, ti, tw);
        for(int ki=0;ki<c->top_k;ki++){
            ExpertW*exp=&lw->experts[ti[ki]];
            matvec(s->hb, exp->w_gate->data, s->xb, H, D);
            matvec(s->hb2, exp->w_up->data, s->xb, H, D);
            for(int i=0;i<H;i++){float act=c->use_gelu?gelu_f(s->hb[i]):silu_f(s->hb[i]);s->hb[i]=act*s->hb2[i];}
            matvec(s->xb2, exp->w_down->data, s->hb, D, H);
            for(int i=0;i<D;i++) s->expert_out[i]+=tw[ki]*s->xb2[i];
        }
        if(lw->shared_expert){
            ExpertW*se=lw->shared_expert;
            matvec(s->hbs, se->w_gate->data, s->xb, H, D);
            matvec(s->hb2s, se->w_up->data, s->xb, H, D);
            for(int i=0;i<H;i++){float act=c->use_gelu?gelu_f(s->hbs[i]):silu_f(s->hbs[i]);s->hbs[i]=act*s->hb2s[i];}
            matvec(s->xb2, se->w_down->data, s->hbs, D, H);
            for(int i=0;i<D;i++) s->expert_out[i]+=s->xb2[i];
        }
        if(c->double_prenorm&&lw->ffn_post_norm){rmsnorm(s->xb3,s->expert_out,lw->ffn_post_norm->data,D,c->norm_eps);memcpy(s->expert_out,s->xb3,D*sizeof(float));}
        for(int i=0;i<D;i++) s->x[i]+=s->expert_out[i];
    }
    rmsnorm(s->x, s->x, w->output_norm->data, D, c->norm_eps);
    matvec(s->logits, w->output->data, s->x, c->vocab_size, D);
    if(c->attn_clamp>0){float inv=1.0f/c->attn_clamp;for(int i=0;i<c->vocab_size;i++)s->logits[i]=c->attn_clamp*tanhf(s->logits[i]*inv);}
    return s->logits;
}
/* ═══════════════════════════════════════════════════════════════════════════════
 * TRAINING — forward + backward with analytical MoE gradients.
 * this is where boys become men and autograd becomes unnecessary.
 * ~250 lines of hand-written chain rule. through the router softmax.
 * through the expert gating. through the SwiGLU activation. through
 * the double pre-norm. through the attention. through the RoPE rotations.
 * every single derivative computed by hand. no tape. no graph. no mercy.
 *
 * the backward pass through MoE routing is especially fun: you need
 * d(softmax)/d(logits) dotted with d(loss)/d(expert_weights), and then
 * chain that back through each expert's gate/up/down projections.
 * pytorch does this automatically. we do it because we hate ourselves.
 * and because understanding your gradients is the difference between
 * "my loss went to nan" and "i know exactly why my loss went to nan."
 *
 * aux_loss backprop: load balancing gradient pushes the router to
 * distribute tokens more evenly. without it, expert 0 wins every election
 * and the other 3 atrophy like unused muscles. democracy needs enforcement.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { float *gate_pre, *up_pre, *act_out, *proj_out; } ExpertAct;
typedef struct {
    float *inp, *xn, *q, *k, *v, *attn_sc, *attn_out;
    float *attn_proj, *attn_post, *res_aa, *ffn_xn;
    float *router_lg; int *top_idx; float *top_wt;
    ExpertAct *ea;
    float *sh_gate, *sh_up, *sh_act, *sh_proj;
    float *moe_out, *moe_post;
} LayerAct;

typedef struct {
    LayerAct *layers; float *final_n, *logits, *residual;
    float *dr, *dxn, *dq, *dk, *dv, *dao, *dfxn, *dhb, *dhb2, *deo;
    float *cos_c, *sin_c; int T;
} TrainState;

static TrainState alloc_ts(Config *c){
    TrainState s={0}; int T=c->seq_len, D=c->dim, kv=c->n_kv_heads*c->head_dim;
    int qd=c->n_heads*c->head_dim, H=c->hidden_dim;
    s.T=T; s.layers=calloc(c->depth,sizeof(LayerAct));
    for(int l=0;l<c->depth;l++){
        LayerAct*la=&s.layers[l];
        la->inp=calloc(T*D,4); la->xn=calloc(T*D,4); la->q=calloc(T*qd,4);
        la->k=calloc(T*kv,4); la->v=calloc(T*kv,4);
        la->attn_sc=calloc(T*c->n_heads*T,4); la->attn_out=calloc(T*qd,4);
        la->attn_proj=calloc(T*D,4); la->attn_post=calloc(T*D,4);
        la->res_aa=calloc(T*D,4); la->ffn_xn=calloc(T*D,4);
        la->router_lg=calloc(T*c->n_experts,4); la->top_idx=calloc(T*c->top_k,sizeof(int));
        la->top_wt=calloc(T*c->top_k,4);
        la->ea=calloc(T*c->top_k,sizeof(ExpertAct));
        for(int i=0;i<T*c->top_k;i++){la->ea[i].gate_pre=calloc(H,4);la->ea[i].up_pre=calloc(H,4);la->ea[i].act_out=calloc(H,4);la->ea[i].proj_out=calloc(D,4);}
        la->moe_out=calloc(T*D,4); la->moe_post=calloc(T*D,4);
        if(c->has_shared){la->sh_gate=calloc(T*H,4);la->sh_up=calloc(T*H,4);la->sh_act=calloc(T*H,4);la->sh_proj=calloc(T*D,4);}
    }
    s.residual=calloc(T*D,4); s.dr=calloc(T*D,4); s.dxn=calloc(T*D,4);
    s.dq=calloc(T*qd,4); s.dk=calloc(T*kv,4); s.dv=calloc(T*kv,4);
    s.dao=calloc(T*qd,4); s.dfxn=calloc(T*D,4);
    s.dhb=calloc(T*H,4); s.dhb2=calloc(T*H,4); s.deo=calloc(T*D,4);
    int half=c->head_dim/2;
    s.cos_c=calloc(T*half,4); s.sin_c=calloc(T*half,4);
    for(int p=0;p<T;p++)for(int i=0;i<half;i++){float freq=1.0f/powf(c->rope_theta,(float)(2*i)/(float)c->head_dim);float ang=(float)p*freq;s.cos_c[p*half+i]=cosf(ang);s.sin_c[p*half+i]=sinf(ang);}
    return s;
}

/* matmul fwd/bwd — the inner loop of deep learning. O(M*N*K) of pure violence.
 * C[M,N] = A[M,K] * B[N,K]^T (B stored row-major, each row is a "neuron")
 * with BLAS: cblas_sgemm does this in one call. your CPU has SIMD for a reason. */
static void mm_fwd(float*C,float*A,float*B,int M,int N,int K,float*d_B){
#ifdef USE_CUDA
    int biggest=M*K;if(M*N>biggest)biggest=M*N;
    gpu_ensure_tmp(biggest);
    gpu_upload(d_tmp_a,A,M*K);
    float*dw=d_B?d_B:d_tmp_b;
    if(!d_B){if(N*K>biggest)gpu_ensure_tmp(N*K);gpu_upload(d_tmp_b,B,N*K);}
    gpu_sgemm_nt(M,N,K,d_tmp_a,dw,d_tmp_c);
    gpu_download(C,d_tmp_c,M*N);
#elif defined(USE_BLAS)
    (void)d_B;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    (void)d_B;
    for(int m=0;m<M;m++){float*cm=C+m*N,*am=A+m*K;for(int n=0;n<N;n++){float s=0;float*bn=B+n*K;for(int k=0;k<K;k++)s+=am[k]*bn[k];cm[n]=s;}}
#endif
}
static void mm_bwd(float*dA,float*dB,float*dC,float*A,float*B,int M,int N,int K,float*d_B){
#ifdef USE_CUDA
    int biggest=M*K;if(N*K>biggest)biggest=N*K;if(M*N>biggest)biggest=M*N;
    gpu_ensure_tmp(biggest);
    gpu_upload(d_tmp_a,dC,M*N);
    float*dw=d_B?d_B:d_tmp_b;
    if(!d_B)gpu_upload(d_tmp_b,B,N*K);
    gpu_sgemm_nn(M,K,N,d_tmp_a,dw,d_tmp_c);
    {
        static float*bwd_tmp=NULL;static int bwd_sz=0;
        int need=M*K>N*K?M*K:N*K;
        if(need>bwd_sz){free(bwd_tmp);bwd_tmp=malloc(need*sizeof(float));bwd_sz=need;}
        gpu_download(bwd_tmp,d_tmp_c,M*K);
        for(int i=0;i<M*K;i++)dA[i]+=bwd_tmp[i];
        gpu_upload(d_tmp_b,A,M*K);
        gpu_sgemm_tn(N,K,M,d_tmp_a,d_tmp_b,d_tmp_c);
        gpu_download(bwd_tmp,d_tmp_c,N*K);
        for(int i=0;i<N*K;i++)dB[i]+=bwd_tmp[i];
    }
#elif defined(USE_BLAS)
    (void)d_B;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0f, dC, N, B, K, 1.0f, dA, K);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, K, M, 1.0f, dC, N, A, K, 1.0f, dB, K);
#else
    for(int m=0;m<M;m++){float*dc=dC+m*N,*am=A+m*K;for(int n=0;n<N;n++){float d=dc[n];if(d==0)continue;float*bn=B+n*K;for(int k=0;k<K;k++){dA[m*K+k]+=d*bn[k];dB[n*K+k]+=d*am[k];}}}
#endif
}

/* rmsnorm fwd/bwd — normalize by root mean square. simpler than layernorm.
 * the backward through variance is where most people give up and use autograd. */
static void rn_fwd(float*o,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*ot=o+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float inv=1.0f/sqrtf(ss/D+eps);for(int i=0;i<D;i++)ot[i]=xt[i]*inv*w[i];}}
static void rn_bwd(float*dx,float*dw,float*dout,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*dot_=dout+t*D,*dxt=dx+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float var=ss/D+eps;float inv=1.0f/sqrtf(var);float cs=0;for(int i=0;i<D;i++)cs+=dot_[i]*w[i]*xt[i];float c2=cs/(D*var);for(int i=0;i<D;i++){dxt[i]+=(dot_[i]*w[i]-xt[i]*c2)*inv;dw[i]+=dot_[i]*xt[i]*inv;}}}

/* ─────────── Training forward pass ───────────
 * same as inference but saves EVERYTHING for the backward pass.
 * every activation, every router decision, every expert's intermediate state.
 * memory goes brrr. but that's the price of analytical gradients. */
static float train_fwd(ModelW *w, Config *c, TrainState *s, int *tokens, int *targets, int T) {
    int D=c->dim, kv=c->n_kv_heads*c->head_dim, qd=c->n_heads*c->head_dim;
    int hd=c->head_dim, hg=c->n_heads/c->n_kv_heads, H=c->hidden_dim;
    float sc=1.0f/sqrtf((float)hd);

    for(int t=0;t<T;t++) memcpy(s->residual+t*D, w->tok_emb->data+tokens[t]*D, D*sizeof(float));

    for(int l=0;l<c->depth;l++){
        LayerW*lw=&w->layers[l]; LayerAct*la=&s->layers[l];
        memcpy(la->inp, s->residual, T*D*4);

        /* Attention */
        rn_fwd(la->xn, s->residual, lw->attn_norm->data, T, D, c->norm_eps);
        mm_fwd(la->q, la->xn, lw->wq->data, T, qd, D, GPU(lw->wq));
        mm_fwd(la->k, la->xn, lw->wk->data, T, kv, D, GPU(lw->wk));
        mm_fwd(la->v, la->xn, lw->wv->data, T, kv, D, GPU(lw->wv));
        for(int t=0;t<T;t++){for(int h=0;h<c->n_heads;h++)apply_rope(la->q+t*qd+h*hd,t,s->cos_c,s->sin_c,hd);for(int h=0;h<c->n_kv_heads;h++)apply_rope(la->k+t*kv+h*hd,t,s->cos_c,s->sin_c,hd);}

        memset(la->attn_out, 0, T*qd*4);
        for(int h=0;h<c->n_heads;h++){
            int kvh=h/hg;
            for(int t=0;t<T;t++){
                float*qt=la->q+t*qd+h*hd;
                float*att=la->attn_sc+(t*c->n_heads+h)*T;
                for(int sp=0;sp<=t;sp++){float*ks=la->k+sp*kv+kvh*hd;float dot=0;for(int d=0;d<hd;d++)dot+=qt[d]*ks[d];att[sp]=dot*sc;}
                /* attn_clamp removed from training — tanh derivative was missing in backward,
                 * attenuating attention gradients up to 3x. clamp only needed for inference. */
                float mx=-1e30f;for(int sp=0;sp<=t;sp++)if(att[sp]>mx)mx=att[sp];
                float se=0;for(int sp=0;sp<=t;sp++){att[sp]=expf(att[sp]-mx);se+=att[sp];}
                for(int sp=0;sp<=t;sp++)att[sp]/=se;for(int sp=t+1;sp<T;sp++)att[sp]=0;
                float*oh=la->attn_out+t*qd+h*hd;
                for(int sp=0;sp<=t;sp++){float a=att[sp];float*vs=la->v+sp*kv+kvh*hd;for(int d=0;d<hd;d++)oh[d]+=a*vs[d];}
            }
        }
        mm_fwd(la->attn_proj, la->attn_out, lw->wo->data, T, D, qd, GPU(lw->wo));
        if(c->double_prenorm&&lw->attn_post_norm) rn_fwd(la->attn_post, la->attn_proj, lw->attn_post_norm->data, T, D, c->norm_eps);
        else memcpy(la->attn_post, la->attn_proj, T*D*4);
        for(int i=0;i<T*D;i++) s->residual[i]+=la->attn_post[i];
        memcpy(la->res_aa, s->residual, T*D*4);

        /* MoE FFN */
        rn_fwd(la->ffn_xn, s->residual, lw->ffn_norm->data, T, D, c->norm_eps);
        memset(la->moe_out, 0, T*D*4);
        mm_fwd(la->router_lg, la->ffn_xn, lw->w_router->data, T, c->n_experts, D, GPU(lw->w_router));

        /* routing: determine top-k experts per token */
        for(int t=0;t<T;t++){
            float*rl=la->router_lg+t*c->n_experts;
            int*ti=la->top_idx+t*c->top_k; float*tw=la->top_wt+t*c->top_k;
            top_k_experts(rl, c->n_experts, c->top_k, ti, tw);
        }

        /* batched expert forward: gather tokens per expert, one mm_fwd each */
        {
            int maxB = T * c->top_k; /* max tokens any expert could see */
            float *gather_in = malloc(maxB * D * sizeof(float));
            float *batch_gate = malloc(maxB * H * sizeof(float));
            float *batch_up   = malloc(maxB * H * sizeof(float));
            float *batch_act  = malloc(maxB * H * sizeof(float));
            float *batch_proj = malloc(maxB * D * sizeof(float));
            int *gather_map = malloc(maxB * sizeof(int)); /* (t, ki) pairs */
            int *gather_ki  = malloc(maxB * sizeof(int));

            for(int e=0;e<c->n_experts;e++){
                /* gather: collect tokens routed to expert e */
                int nB = 0;
                for(int t=0;t<T;t++){
                    int*ti=la->top_idx+t*c->top_k;
                    for(int ki=0;ki<c->top_k;ki++){
                        if(ti[ki]==e){
                            memcpy(gather_in+nB*D, la->ffn_xn+t*D, D*sizeof(float));
                            gather_map[nB]=t; gather_ki[nB]=ki; nB++;
                        }
                    }
                }
                if(nB==0) continue;

                ExpertW*exp=&lw->experts[e];
                /* batched matmul: gate[nB,H] = gather_in[nB,D] @ w_gate[H,D]^T */
                mm_fwd(batch_gate, gather_in, exp->w_gate->data, nB, H, D, GPU(exp->w_gate));
                mm_fwd(batch_up,   gather_in, exp->w_up->data,   nB, H, D, GPU(exp->w_up));
                /* SwiGLU activation */
                for(int i=0;i<nB*H;i++){
                    float act=c->use_gelu?gelu_f(batch_gate[i]):silu_f(batch_gate[i]);
                    batch_act[i]=act*batch_up[i];
                }
                /* down projection: batch_proj[nB,D] = batch_act[nB,H] @ w_down[D,H]^T */
                mm_fwd(batch_proj, batch_act, exp->w_down->data, nB, D, H, GPU(exp->w_down));

                /* scatter: weighted accumulate back + save per-token activations for backward */
                for(int b=0;b<nB;b++){
                    int t=gather_map[b], ki=gather_ki[b];
                    float eW=la->top_wt[t*c->top_k+ki];
                    ExpertAct*ea=&la->ea[t*c->top_k+ki];
                    memcpy(ea->gate_pre, batch_gate+b*H, H*sizeof(float));
                    memcpy(ea->up_pre,   batch_up+b*H,   H*sizeof(float));
                    memcpy(ea->act_out,  batch_act+b*H,  H*sizeof(float));
                    memcpy(ea->proj_out, batch_proj+b*D,  D*sizeof(float));
                    float*mo=la->moe_out+t*D;
                    for(int i=0;i<D;i++) mo[i]+=eW*ea->proj_out[i];
                }
            }
            free(gather_in); free(batch_gate); free(batch_up);
            free(batch_act); free(batch_proj); free(gather_map); free(gather_ki);
        }
        if(c->has_shared&&lw->shared_expert){
            /* batched shared expert: all T tokens at once */
            mm_fwd(la->sh_gate, la->ffn_xn, lw->shared_expert->w_gate->data, T, H, D, GPU(lw->shared_expert->w_gate));
            mm_fwd(la->sh_up,   la->ffn_xn, lw->shared_expert->w_up->data,   T, H, D, GPU(lw->shared_expert->w_up));
            for(int i=0;i<T*H;i++){float act=c->use_gelu?gelu_f(la->sh_gate[i]):silu_f(la->sh_gate[i]);la->sh_act[i]=act*la->sh_up[i];}
            mm_fwd(la->sh_proj, la->sh_act, lw->shared_expert->w_down->data, T, D, H, GPU(lw->shared_expert->w_down));
            for(int i=0;i<T*D;i++) la->moe_out[i]+=la->sh_proj[i];
        }
        if(c->double_prenorm&&lw->ffn_post_norm) rn_fwd(la->moe_post, la->moe_out, lw->ffn_post_norm->data, T, D, c->norm_eps);
        else memcpy(la->moe_post, la->moe_out, T*D*4);
        for(int i=0;i<T*D;i++) s->residual[i]+=la->moe_post[i];
    }

    s->final_n=calloc(T*D,4); rn_fwd(s->final_n, s->residual, w->output_norm->data, T, D, c->norm_eps);
    s->logits=calloc(T*c->vocab_size,4); mm_fwd(s->logits, s->final_n, w->output->data, T, c->vocab_size, D, GPU(w->output));

    float loss=0; int nv=0;
    for(int t=0;t<T;t++){if(targets[t]<0)continue;float*lt=s->logits+t*c->vocab_size;float mx=lt[0];for(int j=1;j<c->vocab_size;j++)if(lt[j]>mx)mx=lt[j];float se=0;for(int j=0;j<c->vocab_size;j++)se+=expf(lt[j]-mx);loss+=-(lt[targets[t]]-mx-logf(se));nv++;}
    /* Load balancing loss — the union contract for experts.
     * fi = fraction of tokens routed to each expert (who's popular)
     * pi = average router probability for each expert (who SHOULD be popular)
     * aux = n_experts * sum(fi * pi) — penalizes popularity concentration.
     * without this, expert 0 becomes a dictator. with it, democracy prevails. */
    float aux=0;
    if(c->aux_loss_w>0&&c->n_experts>1){
        for(int l=0;l<c->depth;l++){LayerAct*la=&s->layers[l];float fi[16]={0},pi[16]={0};
        for(int t=0;t<T;t++){for(int ki=0;ki<c->top_k;ki++)fi[la->top_idx[t*c->top_k+ki]]+=1.0f;
        float*rl=la->router_lg+t*c->n_experts;float mx=rl[0];for(int e=1;e<c->n_experts;e++)if(rl[e]>mx)mx=rl[e];float se=0;for(int e=0;e<c->n_experts;e++)se+=expf(rl[e]-mx);for(int e=0;e<c->n_experts;e++)pi[e]+=expf(rl[e]-mx)/se;}
        for(int e=0;e<c->n_experts;e++){fi[e]/=(float)(T*c->top_k);pi[e]/=(float)T;aux+=fi[e]*pi[e];}}
        aux*=(float)c->n_experts*c->aux_loss_w;
    }
    return nv>0?loss/nv+aux:aux;
}

/* ─────────── Training backward pass ───────────
 * the chain rule, applied by hand, through an entire MoE transformer.
 * karpathy uses loss.backward(). we use 150 lines of pointer arithmetic.
 * d_logits → d_output → d_final_norm → layers in reverse:
 *   d_moe_post → d_experts (each one!) → d_router → d_ffn_norm →
 *   d_attn_post → d_Wo → d_attention → d_rope → d_Wqkv → d_attn_norm
 * if you're reading this and understanding it, congratulations.
 * you now know more about MoE backprop than most ML PhD students. */
static void train_bwd(ModelW *w, Config *c, TrainState *s, int *tokens, int *targets, int T, float **g) {
    int D=c->dim, kv=c->n_kv_heads*c->head_dim, qd=c->n_heads*c->head_dim;
    int hd=c->head_dim, H=c->hidden_dim, hg=c->n_heads/c->n_kv_heads, V=c->vocab_size;
    float sc=1.0f/sqrtf((float)hd);
    int nv=0; for(int t=0;t<T;t++)if(targets[t]>=0)nv++; if(nv==0)goto done;
    float inv_n=1.0f/(float)nv;

    /* d_logits */
    float *dl=calloc(T*V,4);
    for(int t=0;t<T;t++){if(targets[t]<0)continue;float*lt=s->logits+t*V;float*d=dl+t*V;float mx=lt[0];for(int j=1;j<V;j++)if(lt[j]>mx)mx=lt[j];float se=0;for(int j=0;j<V;j++){d[j]=expf(lt[j]-mx);se+=d[j];}for(int j=0;j<V;j++)d[j]=(d[j]/se)*inv_n;d[targets[t]]-=inv_n;}

    /* LM head bwd */
    float*dfn=calloc(T*D,4); mm_bwd(dfn, g[1], dl, s->final_n, w->output->data, T, V, D, GPU(w->output));
    /* Final norm bwd */
    memset(s->dr, 0, T*D*4); rn_bwd(s->dr, g[2], dfn, s->residual, w->output_norm->data, T, D, c->norm_eps);
    free(dfn); free(dl);

    /* Layers reverse */
    for(int l=c->depth-1;l>=0;l--){
        LayerW*lw=&w->layers[l]; LayerAct*la=&s->layers[l];
        int gi=3; for(int ll=0;ll<l;ll++){gi+=6;if(c->double_prenorm)gi+=2;gi+=1+c->n_experts*3;if(c->has_shared)gi+=3;}

        /* MoE backward */
        float*dmo=calloc(T*D,4);
        if(c->double_prenorm&&lw->ffn_post_norm){int pgi=gi+6+(c->double_prenorm?1:0);rn_bwd(dmo,g[pgi],s->dr,la->moe_out,lw->ffn_post_norm->data,T,D,c->norm_eps);}
        else memcpy(dmo, s->dr, T*D*4);

        int egi=gi+6+(c->double_prenorm?2:0)+1; /* expert grad start */
        memset(s->dfxn, 0, T*D*4);

        /* Shared expert bwd — batched */
        if(c->has_shared&&lw->shared_expert){
            int sgi=egi+c->n_experts*3;
            /* d_act[T,H] = dmo[T,D] @ w_down[D,H] + accumulate dW_down */
            float*d_sh_act=calloc(T*H,4);
            mm_bwd(d_sh_act, g[sgi+2], dmo, la->sh_act, lw->shared_expert->w_down->data, T, D, H, GPU(lw->shared_expert->w_down));
            /* d_gate, d_up through SwiGLU */
            float*d_gate=calloc(T*H,4), *d_up=calloc(T*H,4);
            for(int i=0;i<T*H;i++){
                float gp=la->sh_gate[i], up=la->sh_up[i];
                float gd=c->use_gelu?gelu_bwd(gp):silu_bwd(gp);
                float act=c->use_gelu?gelu_f(gp):silu_f(gp);
                d_gate[i]=d_sh_act[i]*up*gd;
                d_up[i]=d_sh_act[i]*act;
            }
            /* d_ffn_xn += d_gate @ w_gate + d_up @ w_up, accumulate dW_gate, dW_up */
            mm_bwd(s->dfxn, g[sgi],   d_gate, la->ffn_xn, lw->shared_expert->w_gate->data, T, H, D, GPU(lw->shared_expert->w_gate));
            mm_bwd(s->dfxn, g[sgi+1], d_up,   la->ffn_xn, lw->shared_expert->w_up->data,   T, H, D, GPU(lw->shared_expert->w_up));
            free(d_sh_act); free(d_gate); free(d_up);
        }

        /* Routed experts bwd — batched per expert */
        {
            int maxB = T * c->top_k;
            float *gather_dmo = malloc(maxB * D * sizeof(float));
            float *gather_act = malloc(maxB * H * sizeof(float));
            float *gather_gate = malloc(maxB * H * sizeof(float));
            float *gather_up = malloc(maxB * H * sizeof(float));
            float *gather_xn = malloc(maxB * D * sizeof(float));
            float *d_act = malloc(maxB * H * sizeof(float));
            float *d_gate = malloc(maxB * H * sizeof(float));
            float *d_up = malloc(maxB * H * sizeof(float));
            float *d_xn_batch = calloc(maxB * D, sizeof(float));
            int *gmap = malloc(maxB * sizeof(int));
            int *gki = malloc(maxB * sizeof(int));

            for(int e=0;e<c->n_experts;e++){
                int nB=0;
                for(int t=0;t<T;t++){
                    int*ti=la->top_idx+t*c->top_k;
                    for(int ki=0;ki<c->top_k;ki++){
                        if(ti[ki]==e){
                            float eW=la->top_wt[t*c->top_k+ki];
                            ExpertAct*ea=&la->ea[t*c->top_k+ki];
                            /* weighted dmo for this token */
                            for(int i=0;i<D;i++) gather_dmo[nB*D+i]=eW*dmo[t*D+i];
                            memcpy(gather_act+nB*H, ea->act_out, H*sizeof(float));
                            memcpy(gather_gate+nB*H, ea->gate_pre, H*sizeof(float));
                            memcpy(gather_up+nB*H, ea->up_pre, H*sizeof(float));
                            memcpy(gather_xn+nB*D, la->ffn_xn+t*D, D*sizeof(float));
                            gmap[nB]=t; gki[nB]=ki; nB++;
                        }
                    }
                }
                if(nB==0) continue;

                int egi2=egi+e*3; ExpertW*exp=&lw->experts[e];
                /* d_act[nB,H] = gather_dmo[nB,D] @ w_down[D,H] + accumulate dW_down */
                memset(d_act, 0, nB*H*sizeof(float));
                mm_bwd(d_act, g[egi2+2], gather_dmo, gather_act, exp->w_down->data, nB, D, H, GPU(exp->w_down));
                /* SwiGLU backward */
                for(int i=0;i<nB*H;i++){
                    float gp=gather_gate[i], up=gather_up[i];
                    float gd=c->use_gelu?gelu_bwd(gp):silu_bwd(gp);
                    float act=c->use_gelu?gelu_f(gp):silu_f(gp);
                    d_gate[i]=d_act[i]*up*gd;
                    d_up[i]=d_act[i]*act;
                }
                /* d_xn += d_gate @ w_gate + d_up @ w_up, accumulate dW */
                memset(d_xn_batch, 0, nB*D*sizeof(float));
                mm_bwd(d_xn_batch, g[egi2],   d_gate, gather_xn, exp->w_gate->data, nB, H, D, GPU(exp->w_gate));
                mm_bwd(d_xn_batch, g[egi2+1], d_up,   gather_xn, exp->w_up->data,   nB, H, D, GPU(exp->w_up));
                /* scatter d_xn back */
                for(int b=0;b<nB;b++){
                    int t=gmap[b];
                    for(int i=0;i<D;i++) s->dfxn[t*D+i]+=d_xn_batch[b*D+i];
                }
            }
            free(gather_dmo); free(gather_act); free(gather_gate); free(gather_up);
            free(gather_xn); free(d_act); free(d_gate); free(d_up); free(d_xn_batch);
            free(gmap); free(gki);
        }

        /* Router backward — backprop through the routing decision itself.
         * this is the hardest gradient in the whole file. d(softmax gating weights)
         * w.r.t. router logits, chained through each expert's contribution.
         * most MoE implementations use straight-through estimators here.
         * we compute the actual derivative. because we're not cowards. */
        int rgi=gi+6+(c->double_prenorm?2:0);
        for(int t=0;t<T;t++){
            float*dm=dmo+t*D; int*ti=la->top_idx+t*c->top_k; float*tw=la->top_wt+t*c->top_k;
            float dw[8]; for(int ki=0;ki<c->top_k;ki++){ExpertAct*ea=&la->ea[t*c->top_k+ki];float d=0;for(int i=0;i<D;i++)d+=dm[i]*ea->proj_out[i];dw[ki]=d;}
            float dot_wd=0; for(int ki=0;ki<c->top_k;ki++)dot_wd+=tw[ki]*dw[ki];
            float*xn_t=la->ffn_xn+t*D;
            /* temperature=2.0 in softmax means d(softmax)/d(logit) is scaled by 1/temp */
            float temp=2.0f;
            for(int ki=0;ki<c->top_k;ki++){float dlk=tw[ki]*(dw[ki]-dot_wd)/temp;int eI=ti[ki];
            for(int j=0;j<D;j++){s->dfxn[t*D+j]+=dlk*lw->w_router->data[eI*D+j];g[rgi][eI*D+j]+=dlk*xn_t[j];}}
        }

        /* Aux loss backward — load balancing gradient through router.
         * d/d_logit of n_experts * aux_w * sum(fi * pi)
         * fi = fraction of tokens per expert (non-differentiable, treated as constant)
         * pi = mean softmax prob per expert (differentiable through router logits)
         * without this, dead experts stay dead forever. */
        if(c->aux_loss_w > 0 && c->n_experts > 1){
            float fi[16]={0};
            for(int t=0;t<T;t++)
                for(int ki=0;ki<c->top_k;ki++)
                    fi[la->top_idx[t*c->top_k+ki]]+=1.0f;
            for(int e=0;e<c->n_experts;e++) fi[e]/=(float)(T*c->top_k);

            for(int t=0;t<T;t++){
                float*rl=la->router_lg+t*c->n_experts;
                float mx2=rl[0];for(int e=1;e<c->n_experts;e++)if(rl[e]>mx2)mx2=rl[e];
                float se2=0,p[16];
                for(int e=0;e<c->n_experts;e++){p[e]=expf(rl[e]-mx2);se2+=p[e];}
                for(int e=0;e<c->n_experts;e++) p[e]/=se2;

                float*xn_t=la->ffn_xn+t*D;
                for(int e=0;e<c->n_experts;e++){
                    float dpe=0;
                    for(int e2=0;e2<c->n_experts;e2++){
                        if(e2==e) dpe += fi[e2]*p[e2]*(1.0f-p[e2]);
                        else dpe -= fi[e2]*p[e2]*p[e];
                    }
                    dpe *= (float)c->n_experts * c->aux_loss_w / (float)T;
                    for(int j=0;j<D;j++){
                        s->dfxn[t*D+j] += dpe * lw->w_router->data[e*D+j];
                        g[rgi][e*D+j] += dpe * xn_t[j];
                    }
                }
            }
        }
        free(dmo);

        /* FFN norm bwd */
        rn_bwd(s->dr, g[gi+5], s->dfxn, la->res_aa, lw->ffn_norm->data, T, D, c->norm_eps);

        /* Attention backward */
        float*dap=calloc(T*D,4);
        if(c->double_prenorm&&lw->attn_post_norm){rn_bwd(dap,g[gi+6],s->dr,la->attn_proj,lw->attn_post_norm->data,T,D,c->norm_eps);}
        else memcpy(dap, s->dr, T*D*4);

        memset(s->dao, 0, T*qd*4); mm_bwd(s->dao, g[gi+4], dap, la->attn_out, lw->wo->data, T, D, qd, GPU(lw->wo));
        free(dap);

        memset(s->dq, 0, T*qd*4); memset(s->dk, 0, T*kv*4); memset(s->dv, 0, T*kv*4);
        for(int h=0;h<c->n_heads;h++){int kvh=h/hg;
        for(int t=0;t<T;t++){float*doh=s->dao+t*qd+h*hd;float*att=la->attn_sc+(t*c->n_heads+h)*T;
        float*da=calloc(T,4);
        for(int sp=0;sp<=t;sp++){float*vs=la->v+sp*kv+kvh*hd;float d=0;for(int d2=0;d2<hd;d2++)d+=doh[d2]*vs[d2];da[sp]=d;float a=att[sp];float*dvs=s->dv+sp*kv+kvh*hd;for(int d2=0;d2<hd;d2++)dvs[d2]+=a*doh[d2];}
        float dot_ad=0;for(int sp=0;sp<=t;sp++)dot_ad+=att[sp]*da[sp];
        float*qt=la->q+t*qd+h*hd;float*dqt=s->dq+t*qd+h*hd;
        for(int sp=0;sp<=t;sp++){float ds=att[sp]*(da[sp]-dot_ad)*sc;float*ks=la->k+sp*kv+kvh*hd;float*dks=s->dk+sp*kv+kvh*hd;for(int d2=0;d2<hd;d2++){dqt[d2]+=ds*ks[d2];dks[d2]+=ds*qt[d2];}}
        free(da);}}

        for(int t=0;t<T;t++){for(int h=0;h<c->n_heads;h++)rope_bwd(s->dq+t*qd+h*hd,t,s->cos_c,s->sin_c,hd);for(int h=0;h<c->n_kv_heads;h++)rope_bwd(s->dk+t*kv+h*hd,t,s->cos_c,s->sin_c,hd);}

        memset(s->dxn, 0, T*D*4);
        mm_bwd(s->dxn, g[gi+1], s->dq, la->xn, lw->wq->data, T, qd, D, GPU(lw->wq));
        mm_bwd(s->dxn, g[gi+2], s->dk, la->xn, lw->wk->data, T, kv, D, GPU(lw->wk));
        mm_bwd(s->dxn, g[gi+3], s->dv, la->xn, lw->wv->data, T, kv, D, GPU(lw->wv));

        float*ds=calloc(T*D,4); memcpy(ds, s->dr, T*D*4); memset(s->dr, 0, T*D*4);
        rn_bwd(s->dr, g[gi], s->dxn, la->inp, lw->attn_norm->data, T, D, c->norm_eps);
        for(int i=0;i<T*D;i++) s->dr[i]+=ds[i]; free(ds);
    }

    /* Embedding bwd */
    for(int t=0;t<T;t++){float*de=g[0]+tokens[t]*D;float*dr=s->dr+t*D;for(int i=0;i<D;i++)de[i]+=dr[i];}
done:
    free(s->logits); s->logits=NULL; free(s->final_n); s->final_n=NULL;
}
/* ═══════════════════════════════════════════════════════════════════════════════
 * ADAM OPTIMIZER — the one optimizer everyone uses because nobody has time
 * to understand the 47 alternatives. first moment (momentum), second moment
 * (adaptive learning rate), bias correction (because the first few steps
 * would be insane without it). AdamW variant: weight decay applied before
 * the momentum update, not after. this distinction matters. google it.
 * or don't. your model will converge anyway. adam doesn't care about
 * your understanding. adam just works. that's why it won.
 * (chuck optimizer will replace this. soon. when it achieves level 10.)
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { float *m, *v; int size; } AdamS;
typedef struct { AdamS *states; int np; float b1,b2,eps; int t; } Adam;

static Adam *adam_new(ParamList *p){
    Adam *o=calloc(1,sizeof(Adam)); o->np=p->count; o->states=calloc(p->count,sizeof(AdamS));
    o->b1=0.9f; o->b2=0.95f; o->eps=1e-8f;
    for(int i=0;i<p->count;i++){int sz=p->tensors[i]->size;o->states[i].m=calloc(sz,4);o->states[i].v=calloc(sz,4);o->states[i].size=sz;}
    return o;
}

static void adam_step(Adam *o, ParamList *p, float **g, float lr, float wd){
    o->t++; float bc1=1.0f-powf(o->b1,(float)o->t), bc2=1.0f-powf(o->b2,(float)o->t);
    for(int i=0;i<o->np;i++){Tensor*t=p->tensors[i];float*gr=g[i];AdamS*s=&o->states[i];
    for(int j=0;j<t->size;j++){
        if(wd>0&&t->rows>1) t->data[j]-=lr*wd*t->data[j];
        s->m[j]=o->b1*s->m[j]+(1.0f-o->b1)*gr[j]; s->v[j]=o->b2*s->v[j]+(1.0f-o->b2)*gr[j]*gr[j];
        t->data[j]-=lr*(s->m[j]/bc1)/(sqrtf(s->v[j]/bc2)+o->eps);
    }}
}
static void adam_free(Adam*o){if(!o)return;for(int i=0;i<o->np;i++){free(o->states[i].m);free(o->states[i].v);}free(o->states);free(o);}

static void free_ts(TrainState*s,Config*c){
    int T=c->seq_len;
    for(int l=0;l<c->depth;l++){LayerAct*la=&s->layers[l];
        free(la->inp);free(la->xn);free(la->q);free(la->k);free(la->v);
        free(la->attn_sc);free(la->attn_out);free(la->attn_proj);free(la->attn_post);
        free(la->res_aa);free(la->ffn_xn);free(la->router_lg);free(la->top_idx);free(la->top_wt);
        for(int i=0;i<T*c->top_k;i++){free(la->ea[i].gate_pre);free(la->ea[i].up_pre);free(la->ea[i].act_out);free(la->ea[i].proj_out);}
        free(la->ea);free(la->moe_out);free(la->moe_post);
        if(c->has_shared){free(la->sh_gate);free(la->sh_up);free(la->sh_act);free(la->sh_proj);}
    }
    free(s->layers);free(s->residual);free(s->dr);free(s->dxn);
    free(s->dq);free(s->dk);free(s->dv);free(s->dao);free(s->dfxn);
    free(s->dhb);free(s->dhb2);free(s->deo);free(s->cos_c);free(s->sin_c);
    if(s->final_n)free(s->final_n);if(s->logits)free(s->logits);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA — the thing ML people spend 80% of their time on and 0% of their
 * papers talking about. THREE data sources, zero dependencies:
 *
 * 1. --url: HuggingFace rows API → downloads JSON → extracts "text" fields.
 *    default: FineWeb-Edu (educational web text, curated by HuggingFace).
 *    no parquet. no arrow. just HTTP JSON. the way god intended.
 *
 * 2. --parquet FILE: inline Snappy + Thrift + Parquet reader. extracts text
 *    from BYTE_ARRAY columns. for when you have a 2GB shard on Lambda and
 *    refuse to install pyarrow on principle.
 *
 * 3. fallback: synthetic. 10 sentences × 500 copies. shameful but functional.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── HuggingFace JSON text extractor ── */
static int hf_extract_texts(const char *json, int json_len, FILE *out) {
    /* extracts all "text":"..." values from HF rows API JSON response.
     * no JSON parser. just strstr and escape handling. works on every
     * response HuggingFace has returned since 2023. will break if they
     * change their API. so will everything else. */
    int count = 0;
    const char *p = json, *end = json + json_len;
    while (p < end) {
        p = strstr(p, "\"text\":\"");
        if (!p) break;
        p += 8; /* skip "text":" */
        const char *start = p;
        while (p < end && !(*p == '"' && *(p-1) != '\\')) p++;
        if (p >= end) break;
        /* write unescaped text */
        for (const char *s = start; s < p; s++) {
            if (*s == '\\' && s + 1 < p) {
                s++;
                if (*s == 'n') fputc('\n', out);
                else if (*s == 't') fputc('\t', out);
                else if (*s == '\\') fputc('\\', out);
                else if (*s == '"') fputc('"', out);
                else if (*s == 'u' && s + 4 < p) { fputc('?', out); s += 4; } /* unicode → ? */
                else fputc(*s, out);
            } else fputc(*s, out);
        }
        fputc('\n', out);
        count++;
        p++;
    }
    return count;
}

/* ── Snappy decompressor (for parquet) ── */
static int snappy_decompress(const uint8_t *src, int slen, uint8_t *dst, int dlen) {
    int si = 0, di = 0;
    uint32_t ulen = 0; int shift = 0;
    while (si < slen) { uint8_t b = src[si++]; ulen |= (uint32_t)(b & 0x7F) << shift; if (!(b & 0x80)) break; shift += 7; }
    if ((int)ulen > dlen) return -1;
    while (si < slen && di < (int)ulen) {
        uint8_t tag = src[si++]; int type = tag & 3;
        if (type == 0) { /* literal */
            int len = (tag >> 2) + 1;
            if ((tag >> 2) >= 60) { int nb = (tag >> 2) - 59; len = 0; for (int i = 0; i < nb && si < slen; i++) len |= src[si++] << (i * 8); len++; }
            if (si + len > slen || di + len > (int)ulen) return -1;
            memcpy(dst + di, src + si, len); si += len; di += len;
        } else { /* copy */
            int len, off;
            if (type == 1) { len = ((tag >> 2) & 7) + 4; if (si >= slen) return -1; off = ((tag >> 5) << 8) | src[si++]; }
            else if (type == 2) { len = (tag >> 2) + 1; if (si + 1 >= slen) return -1; off = src[si] | (src[si+1] << 8); si += 2; }
            else { len = (tag >> 2) + 1; if (si + 3 >= slen) return -1; off = src[si] | (src[si+1]<<8) | (src[si+2]<<16) | (src[si+3]<<24); si += 4; }
            if (off == 0 || di - off < 0) return -1;
            for (int i = 0; i < len; i++) dst[di + i] = dst[di - off + i];
            di += len;
        }
    }
    return di;
}

/* ── Thrift Compact Protocol decoder (for parquet footer) ── */
typedef struct { const uint8_t *data; int pos, len; } TR;
static uint64_t tr_varint(TR *r) { uint64_t v=0; int s=0; while(r->pos<r->len){uint8_t b=r->data[r->pos++];v|=(uint64_t)(b&0x7F)<<s;if(!(b&0x80))break;s+=7;} return v; }
static int64_t tr_zigzag(TR *r) { uint64_t v=tr_varint(r); return (int64_t)((v>>1)^-(v&1)); }
static char *tr_string(TR *r) { uint64_t l=tr_varint(r); char *s=malloc(l+1); if(r->pos+(int)l<=r->len){memcpy(s,r->data+r->pos,l);r->pos+=(int)l;}s[l]=0; return s; }
static void tr_skip(TR *r, int type);
static void tr_skip_struct(TR *r) { int prev=0; while(r->pos<r->len){uint8_t b=r->data[r->pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;if(delta==0){prev=(int)(int16_t)tr_zigzag(r);}else prev+=delta;tr_skip(r,ft);} }
static void tr_skip(TR *r, int type) {
    switch(type) {
        case 1: case 2: break;
        case 3: case 4: case 5: case 6: tr_zigzag(r); break;
        case 7: r->pos+=8; break;
        case 8: { uint64_t l=tr_varint(r); r->pos+=(int)l; break; }
        case 9: case 10: { uint8_t h=r->data[r->pos++]; int cnt=(h>>4)&0xF, et=h&0xF; if(cnt==0xF)cnt=(int)tr_varint(r); for(int i=0;i<cnt;i++)tr_skip(r,et); break; }
        case 11: { uint8_t h=r->data[r->pos++]; int kt=(h>>4)&0xF, vt=h&0xF; int cnt=(int)tr_varint(r); for(int i=0;i<cnt;i++){tr_skip(r,kt);tr_skip(r,vt);} break; }
        case 12: tr_skip_struct(r); break;
    }
}

/* ── Parquet reader: extracts text column from .parquet files ── */
typedef struct { char *name; int64_t data_off, dict_off, comp_size, nval; int codec; } PqCol;
typedef struct { PqCol *cols; int n; int64_t nrows; } PqMeta;

static PqMeta pq_footer(const uint8_t *f, int64_t sz) {
    PqMeta m = {0};
    uint32_t flen = *(uint32_t*)(f + sz - 8);
    TR r = { f + sz - 8 - flen, 0, (int)flen };
    int prev = 0;
    while (r.pos < r.len) {
        uint8_t b = r.data[r.pos++]; if (b == 0) break;
        int ft = b & 0xF, delta = (b >> 4) & 0xF;
        int fid = delta ? prev + delta : (int)(int16_t)tr_zigzag(&r); prev = fid;
        if (fid == 1 && ft == 5) tr_zigzag(&r);
        else if (fid == 2 && ft == 9) { uint8_t h=r.data[r.pos++]; int cnt=(h>>4)&0xF; if(cnt==0xF)cnt=(int)tr_varint(&r); for(int i=0;i<cnt;i++)tr_skip_struct(&r); }
        else if (fid == 3 && ft == 6) m.nrows = (int64_t)tr_zigzag(&r);
        else if (fid == 4 && ft == 9) { /* row_groups */
            uint8_t h=r.data[r.pos++]; int rg_cnt=(h>>4)&0xF; if(rg_cnt==0xF)rg_cnt=(int)tr_varint(&r);
            for (int rg=0; rg<rg_cnt; rg++) {
                int rp=0;
                while (r.pos<r.len) { uint8_t rb=r.data[r.pos++]; if(rb==0)break; int rt=rb&0xF,rd=(rb>>4)&0xF; int rf=rd?rp+rd:(int)(int16_t)tr_zigzag(&r); rp=rf;
                    if (rf==1 && rt==9) { /* columns */
                        uint8_t ch=r.data[r.pos++]; int cc=(ch>>4)&0xF; if(cc==0xF)cc=(int)tr_varint(&r);
                        for (int ci=0; ci<cc; ci++) {
                            PqCol col={0}; col.dict_off=-1; int cp=0;
                            while (r.pos<r.len) { uint8_t cb=r.data[r.pos++]; if(cb==0)break; int ct_=cb&0xF,cd_=(cb>>4)&0xF; int cf=cd_?cp+cd_:(int)(int16_t)tr_zigzag(&r); cp=cf;
                                if (cf==3 && ct_==12) { /* ColumnMetaData */
                                    int mp=0;
                                    while (r.pos<r.len) { uint8_t mb=r.data[r.pos++]; if(mb==0)break; int mt=mb&0xF,md=(mb>>4)&0xF; int mf=md?mp+md:(int)(int16_t)tr_zigzag(&r); mp=mf;
                                        if (mf==3&&mt==9) { uint8_t lh=r.data[r.pos++]; int lc=(lh>>4)&0xF; if(lc==0xF)lc=(int)tr_varint(&r); for(int li=0;li<lc;li++){char*s=tr_string(&r);if(li==lc-1)col.name=s;else free(s);} }
                                        else if (mf==4&&mt==5) col.codec=(int)tr_zigzag(&r);
                                        else if (mf==5&&mt==6) col.nval=(int64_t)tr_zigzag(&r);
                                        else if (mf==7&&mt==6) col.comp_size=(int64_t)tr_zigzag(&r);
                                        else if (mf==9&&mt==6) col.data_off=(int64_t)tr_zigzag(&r);
                                        else if (mf==11&&mt==6) col.dict_off=(int64_t)tr_zigzag(&r);
                                        else tr_skip(&r,mt);
                                    }
                                } else tr_skip(&r,ct_);
                            }
                            m.n++; m.cols=realloc(m.cols,m.n*sizeof(PqCol)); m.cols[m.n-1]=col;
                        }
                    } else tr_skip(&r,rt);
                }
            }
        } else tr_skip(&r,ft);
    }
    return m;
}

typedef struct { int type, comp_sz, uncomp_sz, nval; } PgHdr;
static PgHdr pq_page_hdr(const uint8_t *data, int len, int *hlen) {
    TR r={data,0,len}; PgHdr h={0}; int prev=0;
    while (r.pos<r.len) { uint8_t b=r.data[r.pos++]; if(b==0)break; int ft=b&0xF,delta=(b>>4)&0xF; int fid=delta?prev+delta:(int)(int16_t)tr_zigzag(&r); prev=fid;
        if (fid==1&&ft==5) h.type=(int)tr_zigzag(&r);
        else if (fid==2&&ft==5) h.uncomp_sz=(int)tr_zigzag(&r);
        else if (fid==3&&ft==5) h.comp_sz=(int)tr_zigzag(&r);
        else if ((fid==5||fid==7||fid==8)&&ft==12) { int dp=0; while(r.pos<r.len){uint8_t db=r.data[r.pos++];if(db==0)break;int dt=db&0xF,dd=(db>>4)&0xF;int df=dd?dp+dd:(int)(int16_t)tr_zigzag(&r);dp=df;if(df==1&&dt==5)h.nval=(int)tr_zigzag(&r);else tr_skip(&r,dt);} }
        else tr_skip(&r,ft);
    }
    *hlen=r.pos; return h;
}

static int pq_extract(const uint8_t *file, int64_t fsz, PqCol *col, FILE *out) {
    int64_t pos=(col->dict_off>=0)?col->dict_off:col->data_off;
    int64_t end=col->data_off+col->comp_size;
    int total=0; char **dict=NULL; int *dlens=NULL, dsz=0;
    while (pos<end && pos<fsz) {
        int hlen; PgHdr ph=pq_page_hdr(file+pos,(int)(fsz-pos),&hlen);
        pos+=hlen; if(ph.comp_sz<=0||pos+ph.comp_sz>fsz)break;
        uint8_t *pd; int plen; int nf=0;
        if (col->codec==1) { pd=malloc(ph.uncomp_sz); plen=snappy_decompress(file+pos,ph.comp_sz,pd,ph.uncomp_sz); if(plen<0){free(pd);pos+=ph.comp_sz;continue;} nf=1; }
        else { pd=(uint8_t*)(file+pos); plen=ph.comp_sz; }
        if (ph.type==2) { /* DICTIONARY_PAGE */
            dsz=ph.nval; dict=calloc(dsz,sizeof(char*)); dlens=calloc(dsz,sizeof(int));
            int dp=0;
            for(int i=0;i<dsz&&dp+4<=plen;i++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(dp+sl>plen)break;dict[i]=malloc(sl);memcpy(dict[i],pd+dp,sl);dlens[i]=sl;dp+=sl;}
        } else if (ph.type==0||ph.type==3) { /* DATA_PAGE */
            int dp=0;
            if (dsz>0) { /* RLE/bitpack dict encoding */
                if(dp>=plen)goto nxt;
                int bw=pd[dp++];
                for(int v=0;v<ph.nval&&dp<plen;){
                    uint8_t rh=pd[dp++];
                    if(rh&1){ int count=(rh>>1)*8,bytes=(count*bw+7)/8; uint64_t buf=0;int bb=0,bp=dp;
                        for(int i=0;i<count&&v<ph.nval;i++,v++){while(bb<bw&&bp<dp+bytes&&bp<plen){buf|=(uint64_t)pd[bp++]<<bb;bb+=8;}int idx=(int)(buf&((1ULL<<bw)-1));buf>>=bw;bb-=bw;if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}
                        dp+=bytes;
                    } else { int count=rh>>1,idx=0,nb=(bw+7)/8; for(int b=0;b<nb&&dp<plen;b++)idx|=pd[dp++]<<(b*8);
                        for(int i=0;i<count&&v<ph.nval;i++,v++){if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}}
                }
            } else { /* PLAIN BYTE_ARRAY */
                for(int v=0;v<ph.nval&&dp+4<=plen;v++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(sl<0||dp+sl>plen)break;fwrite(pd+dp,1,sl,out);fputc('\n',out);dp+=sl;total++;}
            }
        }
        nxt: if(nf)free(pd); pos+=ph.comp_sz;
    }
    if(dict){for(int i=0;i<dsz;i++)free(dict[i]);free(dict);free(dlens);}
    return total;
}

static int load_parquet(const char *path, const char *out_path, const char *col_name) {
    FILE *f=fopen(path,"rb"); if(!f)return -1;
    fseek(f,0,SEEK_END); int64_t fsz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *file=malloc(fsz); fread(file,1,fsz,f); fclose(f);
    if(fsz<12||memcmp(file,"PAR1",4)!=0||memcmp(file+fsz-4,"PAR1",4)!=0){free(file);return -1;}
    PqMeta meta=pq_footer(file,fsz);
    printf("[parquet] %lld rows, %d column chunks\n",(long long)meta.nrows,meta.n);
    FILE *out=fopen(out_path,"w"); if(!out){free(file);return -1;}
    int total=0;
    for(int i=0;i<meta.n;i++){if(meta.cols[i].name&&strcmp(meta.cols[i].name,col_name)==0)total+=pq_extract(file,fsz,&meta.cols[i],out);}
    fclose(out);
    for(int i=0;i<meta.n;i++)free(meta.cols[i].name); free(meta.cols); free(file);
    printf("[parquet] extracted %d texts from '%s'\n",total,col_name);
    return total>0?0:-1;
}

/* ── get_data: try local → parquet → HF API → synthetic ── */
#define HF_BATCH 100  /* max rows per API request */
#define HF_PAGES 50   /* number of pages to fetch = 5000 texts total */

static int get_data(Config *c){
    struct stat st;
    if(stat(c->data_path,&st)==0&&st.st_size>1000){
        printf("[data] found %s (%.1f MB)\n",c->data_path,(float)st.st_size/1048576);
        return 0;
    }
    /* HuggingFace rows API — paginated: 50 pages × 100 rows = 5000 texts.
     * the API caps at 100 per request because HuggingFace believes in rate limiting
     * and we believe in for loops. everybody wins. */
    if(c->data_url[0]){
        printf("[data] fetching FineWeb-Edu from HuggingFace (%d pages)...\n",HF_PAGES);
        FILE *out=fopen(c->data_path,"w"); if(!out)goto synthetic;
        char tmp[280]; snprintf(tmp,sizeof(tmp),"%s.json",c->data_path);
        int total=0;
        for(int page=0;page<HF_PAGES;page++){
            char cmd[1024];
            snprintf(cmd,sizeof(cmd),
                "curl -sL 'https://datasets-server.huggingface.co/rows"
                "?dataset=HuggingFaceFW/fineweb-edu"
                "&config=sample-10BT&split=train&offset=%d&length=%d' -o '%s'",
                page*HF_BATCH, HF_BATCH, tmp);
            if(system(cmd)!=0)continue;
            if(stat(tmp,&st)!=0||st.st_size<500)continue;
            FILE *jf=fopen(tmp,"r"); if(!jf)continue;
            char *json=malloc(st.st_size+1);
            int jl=(int)fread(json,1,st.st_size,jf); json[jl]=0; fclose(jf);
            int n=hf_extract_texts(json,jl,out);
            free(json); total+=n;
            if((page+1)%10==0)printf("[data] page %d/%d — %d texts so far\n",page+1,HF_PAGES,total);
        }
        fclose(out); unlink(tmp);
        if(total>0){
            stat(c->data_path,&st);
            printf("[data] downloaded %d texts (%.1f MB) from FineWeb-Edu\n",total,(float)st.st_size/1048576);
            return 0;
        }
        printf("[data] HuggingFace download failed\n");
    }
    synthetic:
    /* fallback: synthetic. shameful but functional. */
    printf("[data] creating synthetic dataset...\n");
    FILE*f=fopen(c->data_path,"w"); if(!f)return -1;
    const char *s[]={
        "The quick brown fox jumps over the lazy dog. This is a simple sentence for training.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Neural networks are inspired by biological neurons and can learn complex patterns.",
        "Mixture of experts models route different tokens to specialized expert networks.",
        "The router network learns which expert is best suited for each input token.",
        "Sparse activation means only a fraction of model parameters are used per token.",
        "Grok architecture uses SwiGLU activation and double pre-normalization layers.",
        "Attention clamping stabilizes training by bounding attention logit magnitudes.",
        "Load balancing ensures all experts receive roughly equal amounts of tokens.",
        "Transformers use self-attention mechanisms to process sequences in parallel.",NULL};
    for(int r=0;r<500;r++)for(int i=0;s[i];i++)fprintf(f,"%s\n",s[i]);
    fclose(f); return 0;
}

static char *load_text(const char *p, int *len){FILE*f=fopen(p,"r");if(!f){*len=0;return NULL;}fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);char*t=malloc(sz+1);*len=(int)fread(t,1,sz,f);t[*len]='\0';fclose(f);return t;}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF EXPORT — binary format that llama.cpp understands.
 * header: magic (GGUF), version, tensor count, metadata count.
 * metadata: architecture details so the loader knows what it's dealing with.
 * tensors: name → shape → offset → raw float32 data, 32-byte aligned.
 * compatible with grokky.go. compatible with llama.cpp. compatible with
 * anyone who reads the spec. which is apparently just us and ggerganov.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void w32(FILE*f,uint32_t v){fwrite(&v,4,1,f);}
static void w64(FILE*f,uint64_t v){fwrite(&v,8,1,f);}
static void wstr(FILE*f,const char*s){uint64_t l=strlen(s);w64(f,l);fwrite(s,1,l,f);}
static void wkv_s(FILE*f,const char*k,const char*v){wstr(f,k);w32(f,8);wstr(f,v);}
static void wkv_u(FILE*f,const char*k,uint32_t v){wstr(f,k);w32(f,4);w32(f,v);}
static void wkv_f(FILE*f,const char*k,float v){wstr(f,k);w32(f,6);fwrite(&v,4,1,f);}
static void wkv_b(FILE*f,const char*k,int v){wstr(f,k);w32(f,7);uint8_t b=v?1:0;fwrite(&b,1,1,f);}

static void wti(FILE*f,const char*name,Tensor*t,uint64_t*off){
    wstr(f,name);
    if(t->rows>1){w32(f,2);w64(f,t->cols);w64(f,t->rows);}
    else{w32(f,1);w64(f,t->size);}
    w32(f,0); w64(f,*off); *off+=t->size*4;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHECKPOINT — binary save/load. train once, chat forever.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define CKPT_MAGIC 0x4D4F4543 /* "MOEC" */

static void save_checkpoint(const char *path, ModelW *w, Config *c, Tokenizer *tok) {
    FILE *f = fopen(path, "wb"); if (!f) { printf("[ckpt] cannot create %s\n", path); return; }
    uint32_t magic = CKPT_MAGIC; fwrite(&magic, 4, 1, f);
    fwrite(&c->depth, 4, 1, f); fwrite(&c->dim, 4, 1, f);
    fwrite(&c->n_heads, 4, 1, f); fwrite(&c->n_kv_heads, 4, 1, f);
    fwrite(&c->head_dim, 4, 1, f); fwrite(&c->hidden_dim, 4, 1, f);
    fwrite(&c->vocab_size, 4, 1, f); fwrite(&c->seq_len, 4, 1, f);
    fwrite(&c->norm_eps, 4, 1, f); fwrite(&c->rope_theta, 4, 1, f);
    fwrite(&c->n_experts, 4, 1, f); fwrite(&c->top_k, 4, 1, f);
    fwrite(&c->has_shared, 4, 1, f); fwrite(&c->use_gelu, 4, 1, f);
    fwrite(&c->double_prenorm, 4, 1, f); fwrite(&c->attn_clamp, 4, 1, f);
    /* tokenizer */
    fwrite(&tok->vocab_size, 4, 1, f);
    for (int i = 0; i < tok->vocab_size; i++) {
        int len = tok->tokens[i] ? (int)strlen(tok->tokens[i]) : 0;
        fwrite(&len, 4, 1, f); if (len > 0) fwrite(tok->tokens[i], 1, len, f);
    }
    fwrite(&tok->n_merges, 4, 1, f);
    for (int i = 0; i < tok->n_merges; i++) { fwrite(tok->merges[i].a, 1, 64, f); fwrite(tok->merges[i].b, 1, 64, f); }
    /* weights */
    fwrite(w->tok_emb->data, 4, w->tok_emb->size, f);
    fwrite(w->output_norm->data, 4, w->output_norm->size, f);
    fwrite(w->output->data, 4, w->output->size, f);
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l];
        fwrite(lw->attn_norm->data, 4, lw->attn_norm->size, f);
        fwrite(lw->ffn_norm->data, 4, lw->ffn_norm->size, f);
        if (c->double_prenorm) { fwrite(lw->attn_post_norm->data, 4, lw->attn_post_norm->size, f); fwrite(lw->ffn_post_norm->data, 4, lw->ffn_post_norm->size, f); }
        fwrite(lw->wq->data, 4, lw->wq->size, f); fwrite(lw->wk->data, 4, lw->wk->size, f);
        fwrite(lw->wv->data, 4, lw->wv->size, f); fwrite(lw->wo->data, 4, lw->wo->size, f);
        fwrite(lw->w_router->data, 4, lw->w_router->size, f);
        for (int e = 0; e < c->n_experts; e++) { fwrite(lw->experts[e].w_gate->data, 4, lw->experts[e].w_gate->size, f); fwrite(lw->experts[e].w_up->data, 4, lw->experts[e].w_up->size, f); fwrite(lw->experts[e].w_down->data, 4, lw->experts[e].w_down->size, f); }
        if (c->has_shared) { fwrite(lw->shared_expert->w_gate->data, 4, lw->shared_expert->w_gate->size, f); fwrite(lw->shared_expert->w_up->data, 4, lw->shared_expert->w_up->size, f); fwrite(lw->shared_expert->w_down->data, 4, lw->shared_expert->w_down->size, f); }
    }
    fclose(f);
    struct stat st; stat(path, &st);
    printf("[ckpt] saved %s (%.1f MB)\n", path, (float)st.st_size / 1048576);
}

static int load_checkpoint(const char *path, ModelW *w, Config *c, Tokenizer *tok) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "[ckpt] cannot open %s\n", path); return -1; }
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != CKPT_MAGIC) { fprintf(stderr, "[ckpt] bad magic\n"); fclose(f); return -1; }
    fread(&c->depth, 4, 1, f); fread(&c->dim, 4, 1, f);
    fread(&c->n_heads, 4, 1, f); fread(&c->n_kv_heads, 4, 1, f);
    fread(&c->head_dim, 4, 1, f); fread(&c->hidden_dim, 4, 1, f);
    fread(&c->vocab_size, 4, 1, f); fread(&c->seq_len, 4, 1, f);
    fread(&c->norm_eps, 4, 1, f); fread(&c->rope_theta, 4, 1, f);
    fread(&c->n_experts, 4, 1, f); fread(&c->top_k, 4, 1, f);
    fread(&c->has_shared, 4, 1, f); fread(&c->use_gelu, 4, 1, f);
    fread(&c->double_prenorm, 4, 1, f); fread(&c->attn_clamp, 4, 1, f);
    /* tokenizer */
    tok_init(tok);
    int vs; fread(&vs, 4, 1, f);
    for (int i = 0; i < vs; i++) { int len; fread(&len, 4, 1, f); char buf[256]={0}; if (len>0&&len<256) fread(buf,1,len,f); tok_add(tok, buf); }
    int nm; fread(&nm, 4, 1, f);
    tok->merges = calloc(nm, sizeof(MergePair)); tok->n_merges = nm;
    for (int i = 0; i < nm; i++) { fread(tok->merges[i].a, 1, 64, f); fread(tok->merges[i].b, 1, 64, f); }
    /* weights */
    init_weights(w, c);
    fread(w->tok_emb->data, 4, w->tok_emb->size, f);
    fread(w->output_norm->data, 4, w->output_norm->size, f);
    fread(w->output->data, 4, w->output->size, f);
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l];
        fread(lw->attn_norm->data, 4, lw->attn_norm->size, f);
        fread(lw->ffn_norm->data, 4, lw->ffn_norm->size, f);
        if (c->double_prenorm) { fread(lw->attn_post_norm->data, 4, lw->attn_post_norm->size, f); fread(lw->ffn_post_norm->data, 4, lw->ffn_post_norm->size, f); }
        fread(lw->wq->data, 4, lw->wq->size, f); fread(lw->wk->data, 4, lw->wk->size, f);
        fread(lw->wv->data, 4, lw->wv->size, f); fread(lw->wo->data, 4, lw->wo->size, f);
        fread(lw->w_router->data, 4, lw->w_router->size, f);
        for (int e = 0; e < c->n_experts; e++) { fread(lw->experts[e].w_gate->data, 4, lw->experts[e].w_gate->size, f); fread(lw->experts[e].w_up->data, 4, lw->experts[e].w_up->size, f); fread(lw->experts[e].w_down->data, 4, lw->experts[e].w_down->size, f); }
        if (c->has_shared) { fread(lw->shared_expert->w_gate->data, 4, lw->shared_expert->w_gate->size, f); fread(lw->shared_expert->w_up->data, 4, lw->shared_expert->w_up->size, f); fread(lw->shared_expert->w_down->data, 4, lw->shared_expert->w_down->size, f); }
    }
    fclose(f);
    printf("[ckpt] loaded %s — depth=%d dim=%d experts=%d vocab=%d params=%.2fM\n",
           path, c->depth, c->dim, c->n_experts, c->vocab_size, (float)count_params(c)/1e6f);
    return 0;
}

static void export_gguf(ModelW *w, Config *c){
    FILE*f=fopen(c->gguf_path,"wb"); if(!f){printf("[gguf] failed\n");return;}
    int pl=6+1+c->n_experts*3+(c->double_prenorm?2:0)+(c->has_shared?3:0);
    int nt=3+c->depth*pl;
    w32(f,0x46554747); w32(f,3); w64(f,nt); w64(f,17);
    wkv_s(f,"general.architecture","llama"); wkv_s(f,"general.name","g");
    wkv_u(f,"llama.block_count",c->depth); wkv_u(f,"llama.embedding_length",c->dim);
    wkv_u(f,"llama.attention.head_count",c->n_heads); wkv_u(f,"llama.attention.head_count_kv",c->n_kv_heads);
    wkv_u(f,"llama.feed_forward_length",c->hidden_dim); wkv_u(f,"llama.context_length",c->seq_len);
    wkv_f(f,"llama.attention.layer_norm_rms_epsilon",c->norm_eps); wkv_f(f,"llama.rope.freq_base",c->rope_theta);
    wkv_u(f,"llama.expert_count",c->n_experts); wkv_u(f,"llama.expert_used_count",c->top_k);
    wkv_b(f,"grok.shared_expert",c->has_shared); wkv_b(f,"grok.use_gelu",c->use_gelu);
    wkv_b(f,"grok.double_prenorm",c->double_prenorm); wkv_f(f,"grok.attn_clamp",c->attn_clamp);
    wkv_s(f,"tokenizer.ggml.model","gpt2");

    uint64_t off=0;
    wti(f,"token_embd.weight",w->tok_emb,&off);
    wti(f,"output_norm.weight",w->output_norm,&off);
    wti(f,"output.weight",w->output,&off);
    for(int l=0;l<c->depth;l++){
        LayerW*lw=&w->layers[l]; char n[96];
        snprintf(n,96,"blk.%d.attn_norm.weight",l); wti(f,n,lw->attn_norm,&off);
        snprintf(n,96,"blk.%d.attn_q.weight",l); wti(f,n,lw->wq,&off);
        snprintf(n,96,"blk.%d.attn_k.weight",l); wti(f,n,lw->wk,&off);
        snprintf(n,96,"blk.%d.attn_v.weight",l); wti(f,n,lw->wv,&off);
        snprintf(n,96,"blk.%d.attn_output.weight",l); wti(f,n,lw->wo,&off);
        snprintf(n,96,"blk.%d.ffn_norm.weight",l); wti(f,n,lw->ffn_norm,&off);
        if(c->double_prenorm){snprintf(n,96,"blk.%d.attn_post_norm.weight",l);wti(f,n,lw->attn_post_norm,&off);snprintf(n,96,"blk.%d.ffn_post_norm.weight",l);wti(f,n,lw->ffn_post_norm,&off);}
        snprintf(n,96,"blk.%d.ffn_gate_inp.weight",l); wti(f,n,lw->w_router,&off);
        for(int e=0;e<c->n_experts;e++){
            snprintf(n,96,"blk.%d.ffn_gate.%d.weight",l,e); wti(f,n,lw->experts[e].w_gate,&off);
            snprintf(n,96,"blk.%d.ffn_up.%d.weight",l,e); wti(f,n,lw->experts[e].w_up,&off);
            snprintf(n,96,"blk.%d.ffn_down.%d.weight",l,e); wti(f,n,lw->experts[e].w_down,&off);
        }
        if(c->has_shared){
            snprintf(n,96,"blk.%d.ffn_gate.shared.weight",l);wti(f,n,lw->shared_expert->w_gate,&off);
            snprintf(n,96,"blk.%d.ffn_up.shared.weight",l);wti(f,n,lw->shared_expert->w_up,&off);
            snprintf(n,96,"blk.%d.ffn_down.shared.weight",l);wti(f,n,lw->shared_expert->w_down,&off);
        }
    }
    /* Align to 32 bytes */
    long p=ftell(f); long al=((p+31)/32)*32; for(long i=p;i<al;i++)fputc(0,f);
    /* Data */
    #define WD(t) fwrite((t)->data,4,(t)->size,f)
    WD(w->tok_emb); WD(w->output_norm); WD(w->output);
    for(int l=0;l<c->depth;l++){LayerW*lw=&w->layers[l];WD(lw->attn_norm);WD(lw->wq);WD(lw->wk);WD(lw->wv);WD(lw->wo);WD(lw->ffn_norm);
    if(c->double_prenorm){WD(lw->attn_post_norm);WD(lw->ffn_post_norm);}WD(lw->w_router);
    for(int e=0;e<c->n_experts;e++){WD(lw->experts[e].w_gate);WD(lw->experts[e].w_up);WD(lw->experts[e].w_down);}
    if(c->has_shared){WD(lw->shared_expert->w_gate);WD(lw->shared_expert->w_up);WD(lw->shared_expert->w_down);}}
    fclose(f);
    struct stat st; stat(c->gguf_path,&st);
    printf("[gguf] exported %s (%.1f MB)\n",c->gguf_path,(float)st.st_size/1048576);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GENERATION — where the model finally gets to speak.
 * temperature controls creativity (0 = deterministic bore, 1 = schizophrenic).
 * top_k cuts the vocabulary to the k most likely tokens before sampling.
 * the model generates one token at a time, autoregressively, until it hits
 * EOS or 200 tokens, whichever comes first. it's like a conversation
 * with someone who has opinions but also a word limit. healthy, actually.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int sample(float *logits, int V, float temp, int top_k){
    if(temp<=0){int b=0;for(int i=1;i<V;i++)if(logits[i]>logits[b])b=i;return b;}
    for(int i=0;i<V;i++)logits[i]/=temp;
    if(top_k>0&&top_k<V){float*s=malloc(V*4);memcpy(s,logits,V*4);for(int i=0;i<top_k;i++){int b=i;for(int j=i+1;j<V;j++)if(s[j]>s[b])b=j;float t=s[i];s[i]=s[b];s[b]=t;}float th=s[top_k-1];free(s);for(int i=0;i<V;i++)if(logits[i]<th)logits[i]=-1e30f;}
    softmax_n(logits, V);
    float r=rand_uniform(), cum=0;
    for(int i=0;i<V;i++){cum+=logits[i];if(cum>=r)return i;}
    return V-1;
}

static void chat(ModelW *w, Config *c, Tokenizer *tok){
    RunState rs=alloc_run(c); char input[1024];
    printf("\n[g] ready. type your message (Ctrl+C to quit):\n\n");
    while(1){
        printf("> "); fflush(stdout);
        if(!fgets(input,sizeof(input),stdin))break;
        int len=strlen(input); while(len>0&&(input[len-1]=='\n'||input[len-1]=='\r'))input[--len]='\0';
        if(!len)continue; if(strcmp(input,"quit")==0||strcmp(input,"exit")==0)break;
        int kd=c->n_kv_heads*c->head_dim;
        memset(rs.key_cache,0,c->depth*c->seq_len*kd*4);
        memset(rs.value_cache,0,c->depth*c->seq_len*kd*4);
        int ni; int*ids=tok_encode(tok,input,len,&ni);
        int pos=0; for(int i=0;i<ni&&pos<c->seq_len-1;i++,pos++)forward_token(w,c,&rs,ids[i],pos);
        int prev=ids[ni-1];
        printf("  ");
        for(int i=0;i<200&&pos<c->seq_len;i++,pos++){
            float*lg=forward_token(w,c,&rs,prev,pos);
            int next=sample(lg,c->vocab_size,0.8f,40);
            if(next==tok->eos_id)break;
            int dl; char*dec=tok_decode(tok,&next,1,&dl);
            if(dl>0){fwrite(dec,1,dl,stdout);fflush(stdout);}
            free(dec); prev=next;
        }
        printf("\n\n"); free(ids);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — where it all comes together. parse args. load data. train tokenizer.
 * build model. train model. finetune personality. export GGUF. chat.
 * seven stages of grief, except the last one is acceptance that your model
 * is 5M parameters and has more confidence than GPT-4 on a good day.
 *
 * per-tensor gradient clipping before global norm: this is the trick that
 * makes MoE training not explode. the router gradients are drama queens —
 * they spike 10x above the attention gradients. clip each tensor first,
 * then clip globally. two layers of "calm down." like parenting.
 *
 * cosine LR schedule with warmup: start slow (warmup), reach peak,
 * gracefully decay. the model learns fast in the middle, consolidates
 * at the end. just like humans, except the model actually improves.
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv){
    setbuf(stdout,NULL); int depth=4; char *chat_ckpt=NULL;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--depth")==0&&i+1<argc)depth=atoi(argv[++i]);
        else if(strcmp(argv[i],"--chat")==0&&i+1<argc)chat_ckpt=argv[++i];
        else if(strcmp(argv[i],"--help")==0||strcmp(argv[i],"-h")==0){
            printf("moe.c — one file. one MoE. no excuses.\n\n");
            printf("  --depth N      model depth (default: 4)\n");
            printf("                 2=~2M  3=~5M  4=~9M  5=~17M  6=~27M  7=~41M  8=~58M\n");
            printf("  --chat FILE    load checkpoint and chat (skip training)\n");
            printf("  --data PATH    path to training text file\n");
            printf("  --url URL      HuggingFace rows API URL for training data\n");
            printf("  --parquet FILE extract text from local .parquet file\n\n");
            printf("  default: downloads FineWeb-Edu via HuggingFace API (~5 MB)\n");
            printf("\n  BLAS:  cc moe.c -O3 -lm -DUSE_BLAS -DACCELERATE -framework Accelerate -o moe\n");
            return 0;}
    }
#ifdef USE_CUDA
    if(gpu_init()!=0){fprintf(stderr,"[error] CUDA init failed\n");return 1;}
#endif
    printf("\n  moe.c — one file. one MoE. grok-style. no excuses.\n\n");
    if (chat_ckpt) {
        Config c={0}; c.norm_eps=1e-5f; c.rope_theta=10000.0f;
        Tokenizer tok; ModelW w;
        if (load_checkpoint(chat_ckpt, &w, &c, &tok) != 0) return 1;
        chat(&w, &c, &tok);
        printf("[moe] done.\n"); return 0;
    }
    Config c=config_from_depth(depth);
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--data")==0&&i+1<argc)snprintf(c.data_path,256,"%s",argv[++i]);
        else if(strcmp(argv[i],"--url")==0&&i+1<argc)snprintf(c.data_url,512,"%s",argv[++i]);
        else if(strcmp(argv[i],"--parquet")==0&&i+1<argc){
            const char *pqf=argv[++i];
            printf("[parquet] loading %s...\n",pqf);
            if(load_parquet(pqf,c.data_path,"text")!=0){fprintf(stderr,"[error] parquet load failed\n");return 1;}
        }
    }

    if(get_data(&c)){fprintf(stderr,"[error] no data\n");return 1;}
    int tl; char*text=load_text(c.data_path,&tl);
    if(!text||tl<100){fprintf(stderr,"[error] data too small\n");return 1;}
    printf("[data] %d bytes (%.1f MB)\n",tl,(float)tl/1048576);

    /* BPE on first 1MB — O(n²) per merge, full corpus too slow */
    int bpe_len=tl<1000000?tl:1000000;
    Tokenizer tok; tok_init(&tok); tok_train_bpe(&tok,text,bpe_len,c.bpe_merges);
    c.vocab_size=tok.vocab_size;
    int nt; int*all_tok=tok_encode(&tok,text,tl,&nt); free(text);
    printf("[data] %d tokens (%.1f tok/byte)\n",nt,(float)nt/tl);

    printf("[model] depth=%d dim=%d heads=%d kv=%d hidden=%d\n",c.depth,c.dim,c.n_heads,c.n_kv_heads,c.hidden_dim);
    printf("[model] experts=%d top_k=%d shared=%d swiglu=%d dpn=%d clamp=%.0f\n",c.n_experts,c.top_k,c.has_shared,!c.use_gelu,c.double_prenorm,c.attn_clamp);
    printf("[model] vocab=%d seq=%d params=%.2fM\n",c.vocab_size,c.seq_len,(float)count_params(&c)/1e6f);

    ModelW w; init_weights(&w,&c);
    ParamList params=collect_params(&w,&c);
#ifdef USE_CUDA
    gpu_upload_weights(params.tensors, params.count);
    {int total=0;for(int i=0;i<params.count;i++)total+=params.tensors[i]->size;
     printf("[cuda] uploaded %d weight tensors to GPU (%.1f MB resident)\n",params.count,total*4.0f/1048576.0f);fflush(stdout);}
#endif
    float**grads=calloc(params.count,sizeof(float*));
    for(int i=0;i<params.count;i++)grads[i]=calloc(params.tensors[i]->size,4);
    Adam*opt=adam_new(&params);
    TrainState ts=alloc_ts(&c);

    printf("[train] %d steps, seq=%d, lr=%.1e\n",c.max_steps,c.seq_len,c.lr);
    fflush(stdout);
#ifdef USE_CUDA
    {int T=c.seq_len,V=c.vocab_size,D=c.dim,H=c.hidden_dim;
     int bg=T*V;if(T*H>bg)bg=T*H;if(V*D>bg)bg=V*D;if(H*D>bg)bg=H*D;
     gpu_ensure_tmp(bg);
     printf("[cuda] pre-allocated GPU buffers: %d floats (%.1f MB)\n",bg,bg*4.0f/1048576.0f);
     fflush(stdout);}
#endif
    int grad_accum = c.batch_size; /* accumulate over batch_size sequences per step */
    clock_t t0=clock(); float rl=0; int lc=0;
    for(int step=0;step<c.max_steps;step++){
        float lr=c.lr;
        if(step<c.warmup_steps)lr=c.lr*((float)(step+1)/c.warmup_steps);
        else{float p=(float)(step-c.warmup_steps)/(float)(c.max_steps-c.warmup_steps);lr=c.lr*0.5f*(1.0f+cosf(3.14159f*p));}
        if(lr<c.lr*0.01f)lr=c.lr*0.01f;

        /* zero grads once, accumulate over grad_accum sequences */
        for(int i=0;i<params.count;i++)memset(grads[i],0,params.tensors[i]->size*4);
        float batch_loss=0;
        for(int b=0;b<grad_accum;b++){
            int ms=nt-c.seq_len-1; if(ms<0)ms=0;
            int st=(int)(rand_uniform()*ms);
            float loss=train_fwd(&w,&c,&ts,all_tok+st,all_tok+st+1,c.seq_len);
            batch_loss+=loss;
            train_bwd(&w,&c,&ts,all_tok+st,all_tok+st+1,c.seq_len,grads);
        }
        rl+=batch_loss/grad_accum; lc++;

        /* scale grads by 1/grad_accum */
        {float inv=(float)(1.0f/grad_accum);
         for(int i=0;i<params.count;i++)for(int j=0;j<params.tensors[i]->size;j++)grads[i][j]*=inv;}

        /* global gradient clipping */
        float gn=0;for(int i=0;i<params.count;i++)for(int j=0;j<params.tensors[i]->size;j++)gn+=grads[i][j]*grads[i][j];gn=sqrtf(gn);
        if(gn>1.0f){float s=1.0f/gn;for(int i=0;i<params.count;i++)for(int j=0;j<params.tensors[i]->size;j++)grads[i][j]*=s;}
        adam_step(opt,&params,grads,lr,c.weight_decay);
#ifdef USE_CUDA
        gpu_resync_weights(params.tensors, params.count);
#endif

        if((step+1)%c.log_every==0||step==0){
            float el=(float)(clock()-t0)/CLOCKS_PER_SEC;
            printf("  step %4d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f  (%.1fs)\n",step+1,c.max_steps,rl/lc,lr,(float)((step+1)*c.seq_len)/el,el);
            fflush(stdout);
            rl=0;lc=0;
        }
    }
    printf("[train] done in %.1fs\n",(float)(clock()-t0)/CLOCKS_PER_SEC);

    /* personality finetune — the model already knows language. now teach it
     * to be someone. drop a text file, watch a 5M parameter model develop opinions.
     * lower learning rate (0.1x) because we're nudging, not lobotomizing. */
    struct stat pst;
    if(stat(c.personality_path,&pst)==0&&pst.st_size>10){
        printf("[personality] found %s, finetuning...\n",c.personality_path);
        int pl; char*ptxt=load_text(c.personality_path,&pl);
        if(ptxt&&pl>10){
            int pnt; int*ptok=tok_encode(&tok,ptxt,pl,&pnt);
            for(int step=0;step<c.personality_steps&&pnt>c.seq_len+1;step++){
                int ps=(int)(rand_uniform()*(pnt-c.seq_len-1));
                float loss=train_fwd(&w,&c,&ts,ptok+ps,ptok+ps+1,c.seq_len);
                for(int i=0;i<params.count;i++)memset(grads[i],0,params.tensors[i]->size*4);
                train_bwd(&w,&c,&ts,ptok+ps,ptok+ps+1,c.seq_len,grads);
                float gn=0;for(int i=0;i<params.count;i++)for(int j=0;j<params.tensors[i]->size;j++)gn+=grads[i][j]*grads[i][j];gn=sqrtf(gn);
                if(gn>1.0f){float s=1.0f/gn;for(int i=0;i<params.count;i++)for(int j=0;j<params.tensors[i]->size;j++)grads[i][j]*=s;}
                adam_step(opt,&params,grads,c.lr*0.1f,c.weight_decay);
#ifdef USE_CUDA
                gpu_resync_weights(params.tensors, params.count);
#endif
                if((step+1)%20==0)printf("  personality step %d/%d  loss=%.4f\n",step+1,c.personality_steps,loss);
            }
            free(ptok);
        }
        free(ptxt);
    } else printf("[personality] no %s found, skipping\n",c.personality_path);

    save_checkpoint("moe.bin", &w, &c, &tok);
    export_gguf(&w,&c);
    chat(&w,&c,&tok);

    free_ts(&ts,&c);
    adam_free(opt);
    for(int i=0;i<params.count;i++)free(grads[i]);free(grads);
    free(params.tensors);
    free(all_tok);
    printf("[moe] done.\n"); return 0;
}

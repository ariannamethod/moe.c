/*
 * moe.c — one file. one grok-style MoE. no excuses.
 *
 * Mixture-of-Experts transformer in a single C file. trains from scratch.
 * 4 experts, top-2 routing, shared expert, SwiGLU, double pre-norm,
 * attention clamping, analytical backprop through the router.
 * exports GGUF. chats. no dependencies.
 *
 * cc moe.c -O3 -lm -lpthread -o moe && ./moe --depth 4
 *
 * sibling of l.c (actually.llama). born from the Arianna Method ecosystem.
 * grok's architecture, opus's code, oleg's spite.
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

/* ═══════════════════════════════════════════════════════════════════
 * CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    int depth, dim, n_heads, n_kv_heads, head_dim, hidden_dim;
    int vocab_size, seq_len;
    float norm_eps, rope_theta;
    int n_experts, top_k, has_shared;
    int use_gelu, double_prenorm;
    float attn_clamp, aux_loss_w;
    float lr, weight_decay;
    int batch_size, max_steps, warmup_steps, log_every, bpe_merges, personality_steps;
    char data_path[256], gguf_path[256], personality_path[256];
} Config;

static Config config_from_depth(int depth) {
    Config c = {0};
    c.depth = depth;
    c.dim = depth * 48;
    c.dim = ((c.dim + 63) / 64) * 64;
    if (c.dim < 192) c.dim = 192;
    if (c.dim > 768) c.dim = 768;
    c.head_dim = 64;
    c.n_heads = c.dim / c.head_dim;
    if (c.n_heads < 1) c.n_heads = 1;
    if (c.dim <= 256) { c.n_kv_heads = c.n_heads; }
    else {
        c.n_kv_heads = c.n_heads / 2;
        if (c.n_kv_heads < 1) c.n_kv_heads = 1;
        while (c.n_heads % c.n_kv_heads != 0 && c.n_kv_heads > 1) c.n_kv_heads--;
    }
    c.n_experts = 4; c.top_k = 2; c.has_shared = 1;
    c.hidden_dim = (int)(c.dim * 1.5f);
    c.hidden_dim = ((c.hidden_dim + 63) / 64) * 64;
    c.seq_len = 256; c.norm_eps = 1e-5f; c.rope_theta = 10000.0f;
    c.use_gelu = 0; c.double_prenorm = 1; c.attn_clamp = 30.0f; c.aux_loss_w = 0.01f; /* SwiGLU > GELU */
    c.lr = 3e-4f; c.batch_size = 4; c.warmup_steps = 100;
    c.weight_decay = 0.01f; c.log_every = 20;
    long pe = 12L*depth*c.dim*c.dim + (long)c.n_experts*3*c.dim*c.hidden_dim*depth;
    c.max_steps = (int)(pe * 6 / (c.batch_size * c.seq_len));
    if (c.max_steps < 200) c.max_steps = 200;
    if (c.max_steps > 2000) c.max_steps = 2000;
    c.bpe_merges = 4000; c.personality_steps = 100;
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

/* ═══════════════════════════════════════════════════════════════════
 * RNG
 * ═══════════════════════════════════════════════════════════════════ */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) { rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
static float rand_uniform(void) { return (float)(rng_next()&0x7FFFFFFF)/(float)0x7FFFFFFF; }
static float rand_normal(void) { float u1=rand_uniform(),u2=rand_uniform(); if(u1<1e-10f)u1=1e-10f; return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }

/* ═══════════════════════════════════════════════════════════════════
 * DYNAMIC ARRAYS + BPE TOKENIZER (from l.c / molequla)
 * ═══════════════════════════════════════════════════════════════════ */
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
/* ═══════════════════════════════════════════════════════════════════
 * TENSOR
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct { float *data; int size, rows, cols; } Tensor;
static Tensor *tnew(int s){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(s,sizeof(float));t->size=s;t->rows=1;t->cols=s;return t;}
static Tensor *tnew2d(int r,int co){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(r*co,sizeof(float));t->size=r*co;t->rows=r;t->cols=co;return t;}
static void tinit(Tensor*t,float std){for(int i=0;i<t->size;i++)t->data[i]=rand_normal()*std;}

/* ═══════════════════════════════════════════════════════════════════
 * MODEL WEIGHTS — Grok MoE
 * ═══════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════
 * MATH OPS
 * ═══════════════════════════════════════════════════════════════════ */
static float gelu_f(float x){float c=0.7978845608f;float inner=c*(x+0.044715f*x*x*x);return 0.5f*x*(1.0f+tanhf(inner));}
static float gelu_bwd(float x){float c=0.7978845608f;float inner=c*(x+0.044715f*x*x*x);float th=tanhf(inner);float s2=1.0f-th*th;float di=c*(1.0f+3.0f*0.044715f*x*x);return 0.5f*(1.0f+th)+0.5f*x*s2*di;}
static float silu_f(float x){return x/(1.0f+expf(-x));}
static float silu_bwd(float x){float s=1.0f/(1.0f+expf(-x));return s+x*s*(1.0f-s);}

static void rmsnorm(float*out,float*x,float*w,int d,float eps){float ss=0;for(int i=0;i<d;i++)ss+=x[i]*x[i];float inv=1.0f/sqrtf(ss/d+eps);for(int i=0;i<d;i++)out[i]=x[i]*inv*w[i];}
static void matvec(float*out,float*W,float*x,int r,int co){for(int i=0;i<r;i++){float s=0;float*row=W+i*co;for(int j=0;j<co;j++)s+=row[j]*x[j];out[i]=s;}}
static void softmax_n(float*x,int n){float mx=x[0];for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}for(int i=0;i<n;i++)x[i]/=s;}

static void apply_rope(float*v,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float x0=v[i],x1=v[i+h];v[i]=x0*cc[off+i]-x1*sc[off+i];v[i+h]=x0*sc[off+i]+x1*cc[off+i];}}
static void rope_bwd(float*dv,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float d0=dv[i],d1=dv[i+h];dv[i]=d0*cc[off+i]+d1*sc[off+i];dv[i+h]=-d0*sc[off+i]+d1*cc[off+i];}}

static void top_k_experts(float*logits,int n,int k,int*idx,float*wts){
    int used[16]={0};
    for(int ki=0;ki<k;ki++){float bv=-1e30f;int bi=0;for(int i=0;i<n;i++)if(!used[i]&&logits[i]>bv){bv=logits[i];bi=i;}idx[ki]=bi;wts[ki]=logits[bi];used[bi]=1;}
    float mx=wts[0];for(int i=1;i<k;i++)if(wts[i]>mx)mx=wts[i];float s=0;for(int i=0;i<k;i++){wts[i]=expf(wts[i]-mx);s+=wts[i];}for(int i=0;i<k;i++)wts[i]/=s;
}

/* ═══════════════════════════════════════════════════════════════════
 * INFERENCE FORWARD — single token with KV cache
 * ═══════════════════════════════════════════════════════════════════ */
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
        /* Attention */
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

        /* MoE FFN */
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
/* ═══════════════════════════════════════════════════════════════════
 * TRAINING — forward + backward with analytical MoE gradients
 * ═══════════════════════════════════════════════════════════════════ */
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

/* matmul fwd/bwd */
static void mm_fwd(float*C,float*A,float*B,int M,int N,int K){for(int m=0;m<M;m++){float*cm=C+m*N,*am=A+m*K;for(int n=0;n<N;n++){float s=0;float*bn=B+n*K;for(int k=0;k<K;k++)s+=am[k]*bn[k];cm[n]=s;}}}
static void mm_bwd(float*dA,float*dB,float*dC,float*A,float*B,int M,int N,int K){for(int m=0;m<M;m++){float*dc=dC+m*N,*am=A+m*K;for(int n=0;n<N;n++){float d=dc[n];if(d==0)continue;float*bn=B+n*K;for(int k=0;k<K;k++){dA[m*K+k]+=d*bn[k];dB[n*K+k]+=d*am[k];}}}}

/* rmsnorm fwd/bwd sequence */
static void rn_fwd(float*o,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*ot=o+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float inv=1.0f/sqrtf(ss/D+eps);for(int i=0;i<D;i++)ot[i]=xt[i]*inv*w[i];}}
static void rn_bwd(float*dx,float*dw,float*dout,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*dot_=dout+t*D,*dxt=dx+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float var=ss/D+eps;float inv=1.0f/sqrtf(var);float cs=0;for(int i=0;i<D;i++)cs+=dot_[i]*w[i]*xt[i];float c2=cs/(D*var);for(int i=0;i<D;i++){dxt[i]+=(dot_[i]*w[i]-xt[i]*c2)*inv;dw[i]+=dot_[i]*xt[i]*inv;}}}

/* ─────────── Training forward pass ─────────── */
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
        mm_fwd(la->q, la->xn, lw->wq->data, T, qd, D);
        mm_fwd(la->k, la->xn, lw->wk->data, T, kv, D);
        mm_fwd(la->v, la->xn, lw->wv->data, T, kv, D);
        for(int t=0;t<T;t++){for(int h=0;h<c->n_heads;h++)apply_rope(la->q+t*qd+h*hd,t,s->cos_c,s->sin_c,hd);for(int h=0;h<c->n_kv_heads;h++)apply_rope(la->k+t*kv+h*hd,t,s->cos_c,s->sin_c,hd);}

        memset(la->attn_out, 0, T*qd*4);
        for(int h=0;h<c->n_heads;h++){
            int kvh=h/hg;
            for(int t=0;t<T;t++){
                float*qt=la->q+t*qd+h*hd;
                float*att=la->attn_sc+(t*c->n_heads+h)*T;
                for(int sp=0;sp<=t;sp++){float*ks=la->k+sp*kv+kvh*hd;float dot=0;for(int d=0;d<hd;d++)dot+=qt[d]*ks[d];att[sp]=dot*sc;}
                if(c->attn_clamp>0){float inv=1.0f/c->attn_clamp;for(int sp=0;sp<=t;sp++)att[sp]=c->attn_clamp*tanhf(att[sp]*inv);}
                float mx=-1e30f;for(int sp=0;sp<=t;sp++)if(att[sp]>mx)mx=att[sp];
                float se=0;for(int sp=0;sp<=t;sp++){att[sp]=expf(att[sp]-mx);se+=att[sp];}
                for(int sp=0;sp<=t;sp++)att[sp]/=se;for(int sp=t+1;sp<T;sp++)att[sp]=0;
                float*oh=la->attn_out+t*qd+h*hd;
                for(int sp=0;sp<=t;sp++){float a=att[sp];float*vs=la->v+sp*kv+kvh*hd;for(int d=0;d<hd;d++)oh[d]+=a*vs[d];}
            }
        }
        mm_fwd(la->attn_proj, la->attn_out, lw->wo->data, T, D, qd);
        if(c->double_prenorm&&lw->attn_post_norm) rn_fwd(la->attn_post, la->attn_proj, lw->attn_post_norm->data, T, D, c->norm_eps);
        else memcpy(la->attn_post, la->attn_proj, T*D*4);
        for(int i=0;i<T*D;i++) s->residual[i]+=la->attn_post[i];
        memcpy(la->res_aa, s->residual, T*D*4);

        /* MoE FFN */
        rn_fwd(la->ffn_xn, s->residual, lw->ffn_norm->data, T, D, c->norm_eps);
        memset(la->moe_out, 0, T*D*4);
        mm_fwd(la->router_lg, la->ffn_xn, lw->w_router->data, T, c->n_experts, D);

        for(int t=0;t<T;t++){
            float*rl=la->router_lg+t*c->n_experts;
            int*ti=la->top_idx+t*c->top_k; float*tw=la->top_wt+t*c->top_k;
            top_k_experts(rl, c->n_experts, c->top_k, ti, tw);
            for(int ki=0;ki<c->top_k;ki++){
                int eI=ti[ki]; float eW=tw[ki]; ExpertW*exp=&lw->experts[eI];
                ExpertAct*ea=&la->ea[t*c->top_k+ki]; float*xn_t=la->ffn_xn+t*D;
                matvec(ea->gate_pre, exp->w_gate->data, xn_t, H, D);
                matvec(ea->up_pre, exp->w_up->data, xn_t, H, D);
                for(int i=0;i<H;i++){float act=c->use_gelu?gelu_f(ea->gate_pre[i]):silu_f(ea->gate_pre[i]);ea->act_out[i]=act*ea->up_pre[i];}
                matvec(ea->proj_out, exp->w_down->data, ea->act_out, D, H);
                float*mo=la->moe_out+t*D; for(int i=0;i<D;i++)mo[i]+=eW*ea->proj_out[i];
            }
        }
        if(c->has_shared&&lw->shared_expert){
            for(int t=0;t<T;t++){
                float*xn_t=la->ffn_xn+t*D;
                float*sg=la->sh_gate+t*H,*su=la->sh_up+t*H,*sa=la->sh_act+t*H,*sp=la->sh_proj+t*D;
                matvec(sg, lw->shared_expert->w_gate->data, xn_t, H, D);
                matvec(su, lw->shared_expert->w_up->data, xn_t, H, D);
                for(int i=0;i<H;i++){float act=c->use_gelu?gelu_f(sg[i]):silu_f(sg[i]);sa[i]=act*su[i];}
                matvec(sp, lw->shared_expert->w_down->data, sa, D, H);
                float*mo=la->moe_out+t*D; for(int i=0;i<D;i++)mo[i]+=sp[i];
            }
        }
        if(c->double_prenorm&&lw->ffn_post_norm) rn_fwd(la->moe_post, la->moe_out, lw->ffn_post_norm->data, T, D, c->norm_eps);
        else memcpy(la->moe_post, la->moe_out, T*D*4);
        for(int i=0;i<T*D;i++) s->residual[i]+=la->moe_post[i];
    }

    s->final_n=calloc(T*D,4); rn_fwd(s->final_n, s->residual, w->output_norm->data, T, D, c->norm_eps);
    s->logits=calloc(T*c->vocab_size,4); mm_fwd(s->logits, s->final_n, w->output->data, T, c->vocab_size, D);

    float loss=0; int nv=0;
    for(int t=0;t<T;t++){if(targets[t]<0)continue;float*lt=s->logits+t*c->vocab_size;float mx=lt[0];for(int j=1;j<c->vocab_size;j++)if(lt[j]>mx)mx=lt[j];float se=0;for(int j=0;j<c->vocab_size;j++)se+=expf(lt[j]-mx);loss+=-(lt[targets[t]]-mx-logf(se));nv++;}
    /* Load balancing loss */
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

/* ─────────── Training backward pass ─────────── */
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
    float*dfn=calloc(T*D,4); mm_bwd(dfn, g[1], dl, s->final_n, w->output->data, T, V, D);
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

        /* Shared expert bwd */
        if(c->has_shared&&lw->shared_expert){
            int sgi=egi+c->n_experts*3;
            for(int t=0;t<T;t++){
                float*dm=dmo+t*D,*xn_t=la->ffn_xn+t*D;
                float*sg=la->sh_gate+t*H,*su=la->sh_up+t*H,*sa=la->sh_act+t*H;
                float*da=calloc(H,4); /* d_act through down */
                for(int i=0;i<D;i++)for(int j=0;j<H;j++){da[j]+=dm[i]*lw->shared_expert->w_down->data[i*H+j];g[sgi+2][i*H+j]+=dm[i]*sa[j];}
                for(int i=0;i<H;i++){float gd=c->use_gelu?gelu_bwd(sg[i]):silu_bwd(sg[i]);float act=c->use_gelu?gelu_f(sg[i]):silu_f(sg[i]);
                float dg=da[i]*su[i]*gd, du=da[i]*act;
                for(int j=0;j<D;j++){s->dfxn[t*D+j]+=dg*lw->shared_expert->w_gate->data[i*D+j]+du*lw->shared_expert->w_up->data[i*D+j];g[sgi][i*D+j]+=dg*xn_t[j];g[sgi+1][i*D+j]+=du*xn_t[j];}}
                free(da);
            }
        }

        /* Routed experts bwd */
        for(int t=0;t<T;t++){
            for(int ki=0;ki<c->top_k;ki++){
                int eI=la->top_idx[t*c->top_k+ki]; float eW=la->top_wt[t*c->top_k+ki];
                ExpertW*exp=&lw->experts[eI]; ExpertAct*ea=&la->ea[t*c->top_k+ki];
                float*dm=dmo+t*D,*xn_t=la->ffn_xn+t*D; int egi2=egi+eI*3;
                float*da=calloc(H,4);
                for(int i=0;i<D;i++){float dp=eW*dm[i];for(int j=0;j<H;j++){da[j]+=dp*exp->w_down->data[i*H+j];g[egi2+2][i*H+j]+=dp*ea->act_out[j];}}
                for(int i=0;i<H;i++){float gp=ea->gate_pre[i],up=ea->up_pre[i];float gd=c->use_gelu?gelu_bwd(gp):silu_bwd(gp);float act=c->use_gelu?gelu_f(gp):silu_f(gp);
                float dg=da[i]*up*gd, du=da[i]*act;
                for(int j=0;j<D;j++){s->dfxn[t*D+j]+=dg*exp->w_gate->data[i*D+j]+du*exp->w_up->data[i*D+j];g[egi2][i*D+j]+=dg*xn_t[j];g[egi2+1][i*D+j]+=du*xn_t[j];}}
                free(da);
            }
        }

        /* Router backward */
        int rgi=gi+6+(c->double_prenorm?2:0);
        for(int t=0;t<T;t++){
            float*dm=dmo+t*D; int*ti=la->top_idx+t*c->top_k; float*tw=la->top_wt+t*c->top_k;
            float dw[8]; for(int ki=0;ki<c->top_k;ki++){ExpertAct*ea=&la->ea[t*c->top_k+ki];float d=0;for(int i=0;i<D;i++)d+=dm[i]*ea->proj_out[i];dw[ki]=d;}
            float dot_wd=0; for(int ki=0;ki<c->top_k;ki++)dot_wd+=tw[ki]*dw[ki];
            float*xn_t=la->ffn_xn+t*D;
            for(int ki=0;ki<c->top_k;ki++){float dlk=tw[ki]*(dw[ki]-dot_wd);int eI=ti[ki];
            for(int j=0;j<D;j++){s->dfxn[t*D+j]+=dlk*lw->w_router->data[eI*D+j];g[rgi][eI*D+j]+=dlk*xn_t[j];}}
        }
        free(dmo);

        /* FFN norm bwd */
        rn_bwd(s->dr, g[gi+5], s->dfxn, la->res_aa, lw->ffn_norm->data, T, D, c->norm_eps);

        /* Attention backward */
        float*dap=calloc(T*D,4);
        if(c->double_prenorm&&lw->attn_post_norm){rn_bwd(dap,g[gi+6],s->dr,la->attn_proj,lw->attn_post_norm->data,T,D,c->norm_eps);}
        else memcpy(dap, s->dr, T*D*4);

        memset(s->dao, 0, T*qd*4); mm_bwd(s->dao, g[gi+4], dap, la->attn_out, lw->wo->data, T, D, qd);
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
        mm_bwd(s->dxn, g[gi+1], s->dq, la->xn, lw->wq->data, T, qd, D);
        mm_bwd(s->dxn, g[gi+2], s->dk, la->xn, lw->wk->data, T, kv, D);
        mm_bwd(s->dxn, g[gi+3], s->dv, la->xn, lw->wv->data, T, kv, D);

        float*ds=calloc(T*D,4); memcpy(ds, s->dr, T*D*4); memset(s->dr, 0, T*D*4);
        rn_bwd(s->dr, g[gi], s->dxn, la->inp, lw->attn_norm->data, T, D, c->norm_eps);
        for(int i=0;i<T*D;i++) s->dr[i]+=ds[i]; free(ds);
    }

    /* Embedding bwd */
    for(int t=0;t<T;t++){float*de=g[0]+tokens[t]*D;float*dr=s->dr+t*D;for(int i=0;i<D;i++)de[i]+=dr[i];}
done:
    free(s->logits); s->logits=NULL; free(s->final_n); s->final_n=NULL;
}
/* ═══════════════════════════════════════════════════════════════════
 * ADAM OPTIMIZER
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct { float *m, *v; int size; } AdamS;
typedef struct { AdamS *states; int np; float b1,b2,eps; int t; } Adam;

static Adam *adam_new(ParamList *p){
    Adam *o=calloc(1,sizeof(Adam)); o->np=p->count; o->states=calloc(p->count,sizeof(AdamS));
    o->b1=0.9f; o->b2=0.999f; o->eps=1e-8f;
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

/* ═══════════════════════════════════════════════════════════════════
 * DATA
 * ═══════════════════════════════════════════════════════════════════ */
static int get_data(Config *c){
    struct stat st; if(stat(c->data_path,&st)==0&&st.st_size>1000){printf("[data] found %s (%.1f MB)\n",c->data_path,(float)st.st_size/1048576);return 0;}
    printf("[data] creating synthetic dataset...\n");
    FILE*f=fopen(c->data_path,"w"); if(!f)return -1;
    const char *s[]={
        "The quick brown fox jumps over the lazy dog. This is a simple sentence for training.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Neural networks are inspired by biological neurons and can learn complex patterns.",
        "Mixture of experts models route different tokens to specialized expert networks.",
        "The router network learns which expert is best suited for each input token.",
        "Sparse activation means only a fraction of model parameters are used per token.",
        "Grok architecture uses GELU activation and double pre-normalization layers.",
        "Attention clamping stabilizes training by bounding attention logit magnitudes.",
        "Load balancing ensures all experts receive roughly equal amounts of tokens.",
        "Transformers use self-attention mechanisms to process sequences in parallel.",NULL};
    for(int r=0;r<500;r++)for(int i=0;s[i];i++)fprintf(f,"%s\n",s[i]);
    fclose(f); return 0;
}

static char *load_text(const char *p, int *len){FILE*f=fopen(p,"r");if(!f){*len=0;return NULL;}fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);char*t=malloc(sz+1);*len=(int)fread(t,1,sz,f);t[*len]='\0';fclose(f);return t;}

/* ═══════════════════════════════════════════════════════════════════
 * GGUF EXPORT — compatible with grokky.go
 * ═══════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════
 * GENERATION
 * ═══════════════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv){
    setbuf(stdout,NULL); int depth=4;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--depth")==0&&i+1<argc)depth=atoi(argv[++i]);
        else if(strcmp(argv[i],"--help")==0||strcmp(argv[i],"-h")==0){printf("moe.c — one file. one MoE. no excuses.\n  --depth N  (2=~2M, 4=~5M, 6=~12M, 8=~25M)\n  --data PATH\n");return 0;}
    }
    printf("\n  moe.c — one file. one MoE. grok-style. no excuses.\n\n");
    Config c=config_from_depth(depth);
    for(int i=1;i<argc;i++)if(strcmp(argv[i],"--data")==0&&i+1<argc)snprintf(c.data_path,256,"%s",argv[i+1]);

    if(get_data(&c)){fprintf(stderr,"[error] no data\n");return 1;}
    int tl; char*text=load_text(c.data_path,&tl);
    if(!text||tl<100){fprintf(stderr,"[error] data too small\n");return 1;}
    printf("[data] %d bytes (%.1f MB)\n",tl,(float)tl/1048576);

    Tokenizer tok; tok_init(&tok); tok_train_bpe(&tok,text,tl,c.bpe_merges);
    c.vocab_size=tok.vocab_size;
    int nt; int*all_tok=tok_encode(&tok,text,tl,&nt); free(text);
    printf("[data] %d tokens (%.1f tok/byte)\n",nt,(float)nt/tl);

    printf("[model] depth=%d dim=%d heads=%d kv=%d hidden=%d\n",c.depth,c.dim,c.n_heads,c.n_kv_heads,c.hidden_dim);
    printf("[model] experts=%d top_k=%d shared=%d swiglu=%d dpn=%d clamp=%.0f\n",c.n_experts,c.top_k,c.has_shared,!c.use_gelu,c.double_prenorm,c.attn_clamp);
    printf("[model] vocab=%d seq=%d params=%.2fM\n",c.vocab_size,c.seq_len,(float)count_params(&c)/1e6f);

    ModelW w; init_weights(&w,&c);
    ParamList params=collect_params(&w,&c);
    float**grads=calloc(params.count,sizeof(float*));
    for(int i=0;i<params.count;i++)grads[i]=calloc(params.tensors[i]->size,4);
    Adam*opt=adam_new(&params);
    TrainState ts=alloc_ts(&c);

    printf("[train] %d steps, seq=%d, lr=%.1e\n",c.max_steps,c.seq_len,c.lr);
    clock_t t0=clock(); float rl=0; int lc=0;
    for(int step=0;step<c.max_steps;step++){
        float lr=c.lr;
        if(step<c.warmup_steps)lr=c.lr*((float)(step+1)/c.warmup_steps);
        else{float p=(float)(step-c.warmup_steps)/(float)(c.max_steps-c.warmup_steps);lr=c.lr*0.5f*(1.0f+cosf(3.14159f*p));}
        if(lr<c.lr*0.01f)lr=c.lr*0.01f;

        int ms=nt-c.seq_len-1; if(ms<0)ms=0;
        int st=(int)(rand_uniform()*ms);
        float loss=train_fwd(&w,&c,&ts,all_tok+st,all_tok+st+1,c.seq_len);
        rl+=loss; lc++;
        for(int i=0;i<params.count;i++)memset(grads[i],0,params.tensors[i]->size*4);
        train_bwd(&w,&c,&ts,all_tok+st,all_tok+st+1,c.seq_len,grads);

        float gn=0;for(int i=0;i<params.count;i++)for(int j=0;j<params.tensors[i]->size;j++)gn+=grads[i][j]*grads[i][j];gn=sqrtf(gn);
        if(gn>1.0f){float s=1.0f/gn;for(int i=0;i<params.count;i++)for(int j=0;j<params.tensors[i]->size;j++)grads[i][j]*=s;}
        adam_step(opt,&params,grads,lr,c.weight_decay);

        if((step+1)%c.log_every==0||step==0){
            float el=(float)(clock()-t0)/CLOCKS_PER_SEC;
            printf("  step %4d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f  (%.1fs)\n",step+1,c.max_steps,rl/lc,lr,(float)((step+1)*c.seq_len)/el,el);
            rl=0;lc=0;
        }
    }
    printf("[train] done in %.1fs\n",(float)(clock()-t0)/CLOCKS_PER_SEC);

    /* personality finetune */
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
                if((step+1)%20==0)printf("  personality step %d/%d  loss=%.4f\n",step+1,c.personality_steps,loss);
            }
            free(ptok);
        }
        free(ptxt);
    } else printf("[personality] no %s found, skipping\n",c.personality_path);

    export_gguf(&w,&c);
    chat(&w,&c,&tok);
    free(all_tok); printf("[moe] done.\n"); return 0;
}

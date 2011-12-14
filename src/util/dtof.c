#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Warning: byteswap.h is an unportable GNU extension!  Don't use! */

static inline unsigned short bswap_16(unsigned short x) {
	  return (x>>8) | (x<<8);
}

static inline unsigned int bswap_32(unsigned int x) {
	  return (bswap_16(x&0xffff)<<16) | (bswap_16(x>>16));
}

static inline unsigned long long bswap_64(unsigned long long x) {
	  return (((unsigned long long)bswap_32(x&0xffffffffull))<<32) |
	  (bswap_32(x>>32));
}

int is_bigendian()
{
	int i = 1;
	char *p = (char *) &i;
	if (p[0] == 1) 
		return 0; // little endian
	else
		return 1; // big endian
}

void double_to_float(FILE *ifp, FILE *ofp)
{
	double d;
	float f;
	if (is_bigendian()) {
		while(fread(&d, sizeof(double), 1, ifp)) {
			f = (float) d;
			fwrite(&f, sizeof(float), 1, ofp);
		}
	} else {  // little endian
		long l;
		int i; 
		while (fread(&l, sizeof(long), 1, ifp)) {
			l = bswap_64(l);
			d = *(double *)&l;
			f = (float) d;
			i = *(int *)&f;
			i = bswap_32(i);
			fwrite(&i, sizeof(int), 1, ofp);
		}
	}
}

void convert_file(char *prog, char *base, char *ext)
{
	char *single_indicator = ".sp";

	char *source = (char *) malloc(strlen(base)+strlen(ext)+1);
	strcpy(source, base);
	strcat(source, ext);

	char *target = (char *) malloc(strlen(base)+strlen(single_indicator)+strlen(ext)+1);
	strcpy(target, base);
	strcat(target, single_indicator);
	strcat(target, ext);

	//if (rename(target, source)) printf("%s: can't rename %s\n", prog, target);

	FILE *in, *out; 
	if ((in = fopen(source, "r")) == NULL) { 
		printf("%s: can't open %s\n", prog, source); 
		return;
	} else { 
		if ((out = fopen(target, "w")) == NULL) { 
			printf("%s: can't open %s\n", prog, target); 
			return;
		}
		double_to_float(in, out);
		fclose(in); 
		fclose(out); 
	}

	if (remove(source)) {
		printf("%s: can't remove %s\n", prog, source);
		return;
	}

	free(target);
	free(source);
}

int main(int argc, char *argv[]) 
{
	char *e_ext = ".E";
	char *h_ext = ".H";
	char *prog = argv[0];
	while(--argc > 0) {
		++argv;
		convert_file(prog, *argv, e_ext);
		convert_file(prog, *argv, h_ext);
	}
	return 0;
}


"""Build eval/real.json: the larger multi-project eval set.

Each question has `expect` (primary file(s)) and `also` (other files a reader would accept as a
correct answer). A hit is any file in expect+also. Split is dev/test by a hash of the question, except
MainLandingPage, whose keyword rules were tuned on it: it is dev-only. Questions were written from file
names and a light skim of each project, before running `lt` on them.
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).parent

# project -> (path under $HOME, [(question, [expect], [also])])
NEW = {
"dentist-lv": ("sites/dentist-lv", [
 ("where is the contact form submission handled", ["src/scripts/contact-form.ts"], ["src/components/ContactForm.astro"]),
 ("which component switches between Latvian and English", ["src/components/LocaleSwitch.astro"], ["src/lib/i18n.ts"]),
 ("where are the English translations kept", ["src/i18n/en.json"], ["src/lib/i18n.ts"]),
 ("where are the site pages and navigation structure defined", ["src/site/ia.ts"], []),
 ("which test checks the mobile menu", ["e2e/factory/behaviors.spec.mjs"], []),
 ("which test catches horizontal overflow on small screens", ["e2e/factory/overflow.spec.mjs"], []),
 ("where is the Latvian price list page", ["src/pages/cenas.astro"], ["src/views/Prices.astro"]),
 ("which check makes sure both languages have the same translation keys", ["e2e/factory/i18n-parity.mjs"], []),
 ("where is the quality gate that runs all the checks", ["e2e/factory/gate.mjs"], ["e2e/factory/gate.json"]),
 ("how is the built site served during the e2e tests", ["e2e/factory/serve.mjs"], ["e2e/factory/playwright.config.mjs", "e2e/factory/_site.mjs"]),
 ("where is the English privacy policy page", ["src/pages/en/privacy.astro"], []),
 ("where do the global styles live", ["src/styles/global.css"], []),
 ("what is the footer component", ["src/components/Footer.astro"], []),
 ("which page is shown for unknown urls", ["src/pages/404.astro"], ["src/pages/en/404.astro", "src/views/NotFound.astro"]),
 ("where are hreflang and canonical links emitted", ["src/layouts/Base.astro"], ["e2e/factory/seo-site.spec.mjs"]),
]),
"jobBoard": ("projects/jobBoard", [
 ("where are login attempts rate limited", ["backend/routes/api.php"], ["backend/tests/Feature/AuthThrottlingTest.php", "backend/app/Http/Controllers/AuthController.php"]),
 ("which test covers login throttling", ["backend/tests/Feature/AuthThrottlingTest.php"], []),
 ("where is job description html sanitized", ["backend/app/Services/HtmlSanitizer.php"], ["backend/app/Services/SanitizedDescription.php", "frontend/lib/sanitize.ts", "frontend/components/SafeHtmlRenderer.tsx"]),
 ("how are job listings ranked", ["backend/app/Services/JobRanker.php"], ["backend/config/ranking.php", "backend/app/Console/Commands/RecomputeJobRanking.php"]),
 ("what scheduled command expires old jobs", ["backend/app/Console/Commands/ExpireJobs.php"], ["backend/routes/console.php"]),
 ("where is the import of jobs from the state employment agency implemented", ["backend/app/Services/Nva/NvaSync.php"], ["backend/app/Console/Commands/SyncNvaListings.php", "backend/app/Services/Nva/NvaClient.php"]),
 ("which test checks imported jobs are closed safely when they disappear at the source", ["backend/tests/Feature/NvaSyncClosingTest.php"], []),
 ("where is schema.org JobPosting structured data generated", ["frontend/lib/job-posting-schema.ts"], ["frontend/app/jobs/[id]/page.tsx"]),
 ("where is the kanban hiring pipeline board", ["frontend/components/PipelineBoard.tsx"], ["frontend/components/PipelineColumn.tsx"]),
 ("how is a candidate cv download restricted to the employer", ["backend/app/Http/Controllers/EmployerApplicationController.php"], ["frontend/app/api/cv/[applicationId]/route.ts", "backend/tests/Feature/CvPrivacyTest.php", "backend/app/Policies/JobApplicationPolicy.php"]),
 ("how does the frontend redirect logged out visitors away from dashboards", ["frontend/proxy.ts"], ["frontend/app/dashboard/employer/layout.tsx", "frontend/app/dashboard/seeker/layout.tsx"]),
 ("where is the newsletter subscription api", ["backend/app/Http/Controllers/SubscriberController.php"], ["backend/app/Http/Requests/StoreSubscriberRequest.php", "frontend/actions/subscribe.ts", "frontend/components/NewsletterSignup.tsx"]),
 ("where are the Latvian translations", ["frontend/locales/lv.json"], ["frontend/lib/i18n.ts"]),
 ("how do I run the backend tests in docker", ["backend/docker/run-tests.sh"], ["Makefile", "docker-compose.yml", "backend/README.md", "README.md"]),
 ("which e2e test covers a new employer posting their first job", ["e2e/tests/employer-new.spec.ts"], ["e2e/helpers/employer.ts"]),
 ("how does an employer redeem a coupon", ["backend/app/Http/Controllers/CouponController.php"], ["backend/app/Models/Coupon.php", "frontend/actions/coupon.ts"]),
 ("where is the salary range form field", ["frontend/components/SalaryRangeFields.tsx"], []),
 ("what deletes old job applications for data retention", ["backend/app/Console/Commands/PruneApplications.php"], ["backend/tests/Feature/RetentionTest.php"]),
 ("which controller lets an admin moderate jobs", ["backend/app/Http/Controllers/Admin/JobController.php"], ["backend/tests/Feature/AdminJobModerationTest.php"]),
 ("where are the api routes declared", ["backend/routes/api.php"], []),
 ("where is the domain vocabulary documented", ["docs/agents/domain.md"], []),
 ("how is text extracted from an uploaded document to prefill the job description", ["frontend/app/api/extract-text/route.ts"], ["frontend/components/DescriptionImport.tsx"]),
 ("where is the job search filter ui", ["frontend/components/JobFilters.tsx"], []),
 ("where is the password reset flow", ["backend/app/Http/Controllers/PasswordResetController.php"], ["frontend/actions/password.ts", "frontend/components/ResetPasswordForm.tsx", "frontend/components/ForgotPasswordForm.tsx"]),
 ("where is the sitemap generated", ["frontend/app/sitemap.ts"], []),
]),
"ventmernieks": ("projects/ventmernieks", [
 ("where are hreflang alternates and sitemap configured", ["astro.config.mjs"], ["src/layouts/BaseLayout.astro", "src/i18n/slugs.ts"]),
 ("where is the cookie consent banner", ["src/components/CookieConsent.astro"], []),
 ("where are the translated url slugs for each language", ["src/i18n/slugs.ts"], []),
 ("where are the ui text translations", ["src/i18n/ui.ts"], ["src/i18n/utils.ts"]),
 ("where is the frequently asked questions content", ["src/lib/faq-data.ts"], ["src/components/pages/FaqHub.astro", "src/components/blocks/Faq.astro"]),
 ("where is the glossary page", ["src/components/pages/Glossary.astro"], ["src/lib/glossary-data.ts", "src/pages/terminu-vardnica.astro"]),
 ("where is json-ld structured data for a service page", ["src/components/ServiceSchema.astro"], ["src/components/StructuredData.astro"]),
 ("what implements the mobile navigation menu", ["src/components/MobileMenu.astro"], ["src/components/Header.astro"]),
 ("where is the language picker", ["src/components/LanguagePicker.astro"], []),
 ("how is the site deployed to cloudflare", ["wrangler.jsonc"], ["README.md"]),
 ("where is the llms.txt file for ai crawlers", ["public/llms.txt"], []),
 ("where is the case study content stored", ["src/lib/case-studies-data.ts"], ["src/components/CaseStudyDetail.astro"]),
 ("where are the russian service pages", ["src/pages/ru/uslugi/[slug].astro"], []),
 ("where do the project conventions for coding agents live", ["agents.md"], []),
 ("where is https forced on the apache server", ["public/.htaccess"], []),
 ("where is the customer reviews section of the homepage", ["src/components/blocks/Reviews.astro"], []),
 ("where are the icons for services defined", ["src/lib/service-icons.ts"], []),
]),
"Brutes": ("projects/ClientDemos/Brutes", [
 ("where is the cookie consent banner", ["src/components/CookieConsent.astro"], ["src/components/pages/CookiePolicyPage.astro"]),
 ("where are the route names for each language defined", ["src/i18n/routes.ts"], ["src/pages/[locale]/[...slug].astro"]),
 ("where is the photo gallery component", ["src/components/BentoGallery.astro"], ["src/content/photos.ts", "src/components/pages/GalleryPage.astro"]),
 ("where is the sauna page", ["src/components/pages/SaunaPage.astro"], []),
 ("how is the site configured for cloudflare workers", ["wrangler.jsonc"], ["astro.config.mjs"]),
 ("where are the redirects", ["public/_redirects"], ["src/pages/index.astro"]),
 ("where are meta tags and the html head defined", ["src/layouts/BaseLayout.astro"], []),
 ("where is the Latvian copy", ["src/i18n/lv.ts"], []),
 ("what generates robots.txt", ["src/pages/robots.txt.ts"], []),
 ("where are the project instructions for the coding agent", ["CLAUDE.md"], ["README.md"]),
 ("where is the parallax band component", ["src/components/ParallaxBand.astro"], []),
 ("where are the statistics numbers shown", ["src/components/Stats.astro"], []),
]),
"onboard-test-1": ("projects/onboard-test-1", [
 ("where is a single blog post rendered", ["src/pages/posts/[slug].astro"], []),
 ("how is the sitemap generated", ["src/pages/sitemap.xml.ts"], ["astro.config.mjs"]),
 ("where is the site title and main config", ["src/data/config.ts"], []),
 ("where are the theme colours defined", ["src/data/theme.ts"], ["tailwind.config.cjs"]),
 ("where are the seo meta tags built", ["src/components/seo/SEOTags.astro"], ["src/layouts/Layout.astro"]),
 ("where is the list of portfolio projects", ["src/data/projects.ts"], ["src/components/ProjectCard.astro"]),
 ("where are dates formatted", ["src/utils/formatDate.ts"], []),
 ("where is the blog content collection schema", ["src/content/config.ts"], []),
 ("what does continuous integration run", [".github/workflows/ci-check.yaml"], []),
 ("where are the social media links", ["src/components/SocialLinks.astro"], []),
 ("which component shows a post preview card", ["src/components/PostCard.astro"], []),
]),
"emailer": ("projects/emailer", [
 ("where is the jwt token verified", ["src/middlewares/auth.middleware.js"], []),
 ("how do I generate an api token for a client", ["scripts/generateToken.js"], []),
 ("where is rate limiting configured", ["src/versions/v1/index.js"], []),
 ("where is the send email endpoint implemented", ["src/versions/v1/services/email/email.controller.js"], []),
 ("where is the openapi request schema", ["src/versions/v1/services/email/email.schema.yaml"], []),
 ("where are cors allowed origins set", ["src/app.js"], [".env.example"]),
]),
"javaBootCamp": ("projects/javaBootCamp", [
 ("which class converts inches to centimeters", ["KeywordsAndExpressions/src/ChallangeTwo.java"], []),
 ("where is the nato phonetic alphabet", ["KeywordsAndExpressions/src/ChallangeTwo.java"], []),
 ("where is the high score position calculated", ["KeywordsAndExpressions/src/MainChallenge.java"], []),
 ("which class formats a duration from seconds and minutes", ["KeywordsAndExpressions/src/ChallengeThree.java"], []),
 ("which code prints years and days from minutes", ["KeywordsAndExpressions/src/ChallengeThree.java"], []),
]),
}

def split_of(project: str, q: str) -> str:
    if project == "MainLandingPage":
        return "dev"  # keyword rules were tuned on it
    return "dev" if int(hashlib.sha1(q.encode()).hexdigest(), 16) % 2 == 0 else "test"

out = []
for name, (path, qs) in NEW.items():
    for q, expect, also in qs:
        out.append({"project": name, "path": path, "q": q, "expect": expect, "also": also, "split": split_of(name, q)})
for name, path, f in [("MainLandingPage", "projects/MainLandingPage", "mainlandingpage.json"), ("mireglass", "sites/mireglass", "mireglass.json")]:
    for item in json.loads((HERE / f).read_text())["questions"]:
        out.append({"project": name, "path": path, "q": item["q"], "expect": item["expect"], "also": [], "split": split_of(name, item["q"])})
(HERE / "real.json").write_text(json.dumps({"questions": out}, indent=1) + "\n")
from collections import Counter
print(len(out), Counter(o["split"] for o in out))
